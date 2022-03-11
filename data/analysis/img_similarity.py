import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path

import pytesseract
import yaml
from tqdm import tqdm
import cv2
from glob import glob
from PIL import Image
import torch
import numpy as np
from decord import VideoReader, cpu
from brisque import BRISQUE
import traceback
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
print('USING DEVICE: ' + device)
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training


def encode_images(photos_batch):
    photos = [Image.open(photo_file) for photo_file in photos_batch]
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)

    with torch.no_grad():
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)
    return photos_features.cpu().numpy()


# train = json.load(open('train_data.json', 'r'))
# valid = json.load(open('valid_data.json', 'r'))
# test = json.load(open('test_data.json', 'r'))
#
# dataset = train | valid | test

dataset = yaml.load(open('ann_valid_data_rich.yaml', 'r'))
total_images = 0
diffs_images = 0
total_vids = 0
diffs_vids = 0

for img_set, val in tqdm(list(dataset.items())):
    img_files = list(((Path('games')/ img_set).glob("*.jpg")))
    img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
    img_embs = encode_images(img_files)
    for idx, details in val.items():
        example_diff = 0
        for j in range(10):
            if j != int(idx):
                dist = float(np.linalg.norm(img_embs[int(idx)] - img_embs[j]))
                example_diff += dist
                if 'open-images' in img_set:
                    diffs_images += dist
                    total_images += 1
                else:
                    diffs_vids += dist
                    total_vids += 1
        details['sum_image_differences'] = str(round(example_diff,4))
        dataset[img_set][idx] = details

yaml.dump(dataset, open('ann_valid_data_rich.yaml', 'w'), default_style='"', sort_keys=False)

print(f'Average video similarity: {round(diffs_vids/total_vids, 4)}')
print(f'Average image similarity: {round(diffs_images/total_images, 4)}')