import torch
import os
import json
import sys
from PIL import Image
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image


if len(sys.argv) != 5:
    print("Usage:")
    print("python3 edit.py <instruction.json> <data.json> <input_dir> <output_dir>")
    exit(1)

instruction_json = sys.argv[1]
data_json = sys.argv[2]
input_dir = sys.argv[3]
output_dir = sys.argv[4]

os.makedirs(output_dir, exist_ok=True)

with open(instruction_json, "r", encoding="gbk") as f:
    instructions = json.load(f)

with open(data_json, "r", encoding="utf-8") as f:
    data_list = json.load(f) 


pipe = FluxKontextPipeline.from_pretrained(
    "FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")


for idx, filename in enumerate(data_list):
    input_path = os.path.join(input_dir, filename)

    if not os.path.exists(input_path):
        print("Skip (not exists):", input_path)
        continue

    print(f"[{idx}] Processing:", input_path)

    input_image = load_image(input_path)
    w, h = input_image.size

    new_size = max(w, h)
    padded_image = Image.new("RGB", (new_size, new_size), (255, 255, 255))
    left = (new_size - w) // 2
    top = (new_size - h) // 2
    padded_image.paste(input_image, (left, top))

    image = pipe(
        image=padded_image,
        prompt=instructions[idx % len(instructions)],
        guidance_scale=2.5,
        num_inference_steps=7
    ).images[0]

    out_w, out_h = image.size

    if w >= h:
        crop_h = int(out_w * (h / w))
        top_crop = (out_h - crop_h) // 2
        final_image = image.crop((0, top_crop, out_w, top_crop + crop_h))
    else:
        crop_w = int(out_h * (w / h))
        left_crop = (out_w - crop_w) // 2
        final_image = image.crop((left_crop, 0, left_crop + crop_w, out_h))

    save_path = os.path.join(output_dir, filename)
    final_image.save(save_path)
    print("Saved:", save_path)

print("All done!")
