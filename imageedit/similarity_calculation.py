import torch
import clip
from PIL import Image
import glob
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

def load_and_encode_image(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        image_features = model.encode_image(image_input)     
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
    return image_features

def image_similarity(img_path1: str, img_path2: str) -> float:
    feat1 = load_and_encode_image(img_path1)
    feat2 = load_and_encode_image(img_path2)

    sim = (feat1 @ feat2.T).item()
    return sim

if __name__ == "__main__":
    img1 = ""
    img2 = ""
    similarity = image_similarity(img1, img2)
    print(f"Similarity between {os.path.basename(img1)} and {os.path.basename(img2)}: {similarity:.4f}")
