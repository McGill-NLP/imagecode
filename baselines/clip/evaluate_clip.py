# based on: https://github.com/haltakov/natural-language-image-search#on-your-machine
from tqdm import tqdm
import json
from collections import defaultdict
from glob import glob
import os
import numpy as np
import clip
import torch
from PIL import Image
from pathlib import Path
import statistics
import argparse

def encode_images(photos_batch):
    photos = [Image.open(photo_file) for photo_file in photos_batch]
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)

    with torch.no_grad():
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)
    return photos_features.cpu().numpy()


def encode_text(search_query):
    with torch.no_grad():
        text_encoded = model.encode_text(clip.tokenize(search_query, truncate=True).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    return text_encoded.cpu().numpy()


def find_best_matches(text_features, photo_features):
    similarities = (photo_features @ text_features.T).squeeze(1)
    best_photo_idx = (-similarities).argsort()
    similarities = -similarities
    similarities.sort()
    return best_photo_idx, similarities

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--test_descr_path', type=str, default='../../data/test_data.json')
parser.add_argument('--imgs_path', type=str, default='/network/scratch/b/benno.krojer/dataset/games')
parser.add_argument("--job_id")

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'USING DEVICE: {device}')
model, preprocess = clip.load('ViT-B/16', device=device, jit=False)  # Must set jit=False for training

checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])
print(checkpoint['epoch'])
clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16
model.eval()

correct = 0
total = 0
vid_correct = 0
vid_total = 0
img_correct= 0
img_total = 0


img_dirs = args.imgs_path
descriptions = json.load(open(args.test_descr_path, 'r'))
valid = []
for img_dir, data in descriptions.items():
    for img_idx, text in data.items():
        valid.append((img_dir, img_idx, text))

results = defaultdict(dict)
for img_dir, img_idx, text in tqdm(valid):
    text = [text]
    img_idx = int(img_idx)
    img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
    img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))


    images = [Image.open(photo_file) for photo_file in img_files]
    images = torch.stack([preprocess(photo) for photo in images]).to(device)
    text = clip.tokenize(text, truncate=True).to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = (image_features @ text_features.T).squeeze(1)

    pred = torch.argmax(logits).squeeze()
    if img_idx == pred:
        correct += 1
    if 'open-images' in img_dir:
        img_total += 1
        if img_idx == pred:
            img_correct += 1
    else:
        vid_total += 1
        if img_idx == pred:
            vid_correct += 1        


print('OVERALL ACC: ' + str(round(correct/len(valid),4)))
print('VIDEO ACC: ' + str(round(vid_correct/vid_total,4)))
print('IMG ACC: ' + str(round(img_correct/img_total,4)))
json.dump(results, open(f'results/nocontra-test-data.json', 'w'), indent=2)
