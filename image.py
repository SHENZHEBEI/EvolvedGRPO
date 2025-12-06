import requests
import json
import base64
import os
import random
import dashscope
from http import HTTPStatus
from dashscope import Generation
import openai
from openai import OpenAI
import re
import argparse
import json
import os
import random
import time
from PIL import Image

from mathruler.grader import extract_boxed_content, grade_answer
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import pandas as pd
from datasets import Dataset
def load_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()

def evaluate_chat_model():
    random.seed(0)
    inputs = []
    data_path = ""
    with open(data_path + "data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx in range(2000):
        img_path = data_path + data[idx]["image"]
        print(img_path)
        img = load_image(img_path)
        prompt = ""
        messages = [
            {  
                "role": "user",
                "content": [
                    {"type": "image","image": img_path},
                    {"type": "text", "text": prompt}
                ],
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)

        inputs.append(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_data,  
                }
            }
        )

            
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop_token_ids=stop_token_ids)
    model_outputs = llm.generate(inputs, sampling_params=sampling_params)
    outputs = []
    for data_item, model_output in zip(data, model_outputs):
        print(model_output.outputs[0].text)
        outputs.append(extract_boxed_content(model_output.outputs[0].text))

    output_path = "image_instruction.json"
    json.dump(outputs, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="/home2/yqf/new_dataset/nips/EasyQ1/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--datasets", type=str, default="K12")
parser.add_argument("--out-dir", type=str, default="text_instruction.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

llm = LLM(
    model=args.checkpoint,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8
)
processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
stop_token_ids = None
evaluate_chat_model()
