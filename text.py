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

from mathruler.grader import extract_boxed_content, grade_answer
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import pandas as pd
from datasets import Dataset

def evaluate_chat_model(data):
    random.seed(0)
    inputs = []
    for idx in range(2000):
        messages = [
            {  
                "role": "user",
                "content": [
                    {"type": "text", "text": data[idx]["question"]}
                ],
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs.append(
            {
                "prompt": prompt,
            }
        )

            
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop_token_ids=stop_token_ids)
    model_outputs = llm.generate(inputs, sampling_params=sampling_params)
    outputs = []
    for data_item, model_output in zip(data, model_outputs):
        print(model_output.outputs[0].text)
        match = re.search(r'\\boxed\{(.*?)(?=\})', model_output.outputs[0].text)        
        if match:
            outputs.append(extract_boxed_content(model_output.outputs[0].text))

    output_path = "function.json"
    json.dump(outputs, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    


file_path = 'train.jsonl'

data = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))
print(len(data))
for i in range(2000):
    prompt = data[i]["prompt"]
    match = re.search(r'1\..*?(?=4\.)', prompt, re.DOTALL)

    if match:
        data[i]["question"] = "" + match.group(0).strip()
    else:
        exit(0)    


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="question_generator")
parser.add_argument("--datasets", type=str, default="K12")
parser.add_argument("--out-dir", type=str, default="text_instruction.json")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

llm = LLM(
    model=args.checkpoint,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
)
processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
stop_token_ids = None
evaluate_chat_model(data)
