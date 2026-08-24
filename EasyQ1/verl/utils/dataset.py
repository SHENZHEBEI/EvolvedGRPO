# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import json
import numpy as np
import torch
from datasets import Image as HFImage
from datasets import Dataset as HFDataset
from datasets import Sequence as HFSequence
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts

        path = os.path.join(data_path, "data.json")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if not isinstance(obj, list) or not obj:
            raise ValueError(f"{path} must contain a non-empty JSON list.")
        column_names = {"id", self.prompt_key, self.answer_key, self.image_key}
        if len(column_names) != 4:
            raise ValueError(
                "Dataset column names must be distinct: "
                f"id, {self.prompt_key}, {self.answer_key}, {self.image_key}."
            )
        required_source_keys = ("id", self.prompt_key, self.answer_key, "image")
        for row_index, row in enumerate(obj):
            if not isinstance(row, dict):
                raise ValueError(f"Row {row_index} in {path} is not a JSON object.")
            missing_keys = [key for key in required_source_keys if key not in row]
            if missing_keys:
                raise KeyError(f"Row {row_index} in {path} is missing keys: {missing_keys}")

        if self.processor is None:
            raise ValueError(
                "This dataset contains images, but the configured model does not provide a multimodal processor. "
                "Use a vision-language checkpoint."
            )

        image_paths = [os.path.join(data_path, d["image"]) for d in obj]
        missing_image_paths = [image_path for image_path in image_paths if not os.path.isfile(image_path)]
        if missing_image_paths:
            raise FileNotFoundError(
                f"{len(missing_image_paths)} image files referenced by {path} do not exist. "
                f"First missing image: {missing_image_paths[0]}"
            )

        # Build all columns in one Arrow table. Removing the sole placeholder
        # column from a Hugging Face Dataset also removes its row cardinality,
        # so the previous create/remove/add sequence produced a zero-row table
        # and failed as soon as it tried to append the real id column.
        self.dataset = HFDataset.from_dict(
            {
                "id": [d["id"] for d in obj],
                self.prompt_key: [d[self.prompt_key] for d in obj],
                self.answer_key: [d[self.answer_key] for d in obj],
                self.image_key: [[image_path] for image_path in image_paths],
            }
        )
        self.dataset = self.dataset.cast_column(self.image_key, HFSequence(HFImage()))
        print(f"Loaded {len(self.dataset)} multimodal rows from {path}.")

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if self.filter_overlong_prompts:
            rows_before_filter = len(self.dataset)
            self.dataset = self.dataset.filter(self._filter_overlong_prompts, desc="Filtering overlong prompts")
            removed_rows = rows_before_filter - len(self.dataset)
            print(
                f"Prompt-length validation retained {len(self.dataset)}/{rows_before_filter} rows "
                f"and removed {removed_rows}."
            )

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key not in example:
            return [{"role": "user", "content": prompt_str}]

        images = example[self.image_key]
        if not isinstance(images, (list, tuple)):
            images = [images]
        if not images:
            raise ValueError(f"Example contains an empty '{self.image_key}' field.")

        content_list = []
        if "<image>" in prompt_str:
            prompt_parts = prompt_str.split("<image>")
            placeholder_count = len(prompt_parts) - 1
            if placeholder_count != len(images):
                raise ValueError(
                    f"Prompt contains {placeholder_count} <image> placeholders, "
                    f"but '{self.image_key}' contains {len(images)} images."
                )

            for i, content in enumerate(prompt_parts):
                if i != 0:
                    content_list.append({"type": "image"})
                if content:
                    content_list.append({"type": "text", "text": content})
        else:
            content_list.extend({"type": "image"} for _ in images)
            content_list.append({"type": "text", "text": prompt_str})

        return [{"role": "user", "content": content_list}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        processing_class = self.processor if self.processor is not None else self.tokenizer
        return (
            len(processing_class.apply_chat_template(messages, add_generation_prompt=True)) <= self.max_prompt_length
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            image_values = example.pop(self.image_key)
            if not isinstance(image_values, (list, tuple)):
                image_values = [image_values]
            images = [self.process_image(image) for image in image_values]
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"image": images}
            example["multi_modal_inputs"] = dict(model_inputs)
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and model_inputs.get("image_grid_thw") is not None:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example
