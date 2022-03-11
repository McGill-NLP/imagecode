# inspired from: https://github.com/openai/CLIP/issues/83
# https://github.com/openai/CLIP/issues/83
import json
import os
import random
import wandb
import clip
from clip import model
import torch
from torch import autograd
import tqdm
from torch import nn, optim
from PIL import Image
from pathlib import Path
from collections import defaultdict
import sys
from volta_src.config import BertConfig
from volta_src.embeddings import BertLayerNorm
from volta_src.encoders import GeLU
from extras import convert_sents_to_features, BertLayer
import argparse

random.seed(10)
torch.manual_seed(10)
wandb.init(project='contextualclip', notes="fixed pos emb", entity='bennokrojer', settings=wandb.Settings(start_method="fork"))


def find_best_matches(text_features, photo_features):
    similarities = (photo_features @ text_features.T).squeeze(1)
    best_photo_idx = (-similarities).argsort()
    similarities = -similarities
    similarities.sort()
    return best_photo_idx, similarities


def convert_models_to_fp32(model):
    for p in model.parameters():
        if p.grad is not None:
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()

class ContextualCLIP(torch.nn.Module):
    def __init__(self, bert_config, args):
        super(ContextualCLIP, self).__init__()
        self.clip, self.preprocess = clip.load('ViT-B/16', device=device, jit=False)
        config = BertConfig.from_dict(bert_config)
        self.fusion = args.fusion
        if self.fusion == 'concat':
            hidden_size = 1024
        else:
            hidden_size = 512

        config.hidden_size =  hidden_size
        config.num_attention_heads = 8
        self.transformer = nn.ModuleList([BertLayer(config) for _ in range(args.transformer_layers)])
        self.transformer.cuda()
        self.prediction_layer = nn.Linear(config.hidden_size, 1).cuda()
        self.batch_size = 1
        self.logit_scale = float(args.logit_scale)
        self.frozen_clip = args.frozen_clip
        self.add_input = args.add_input
        self.positional = args.positional
        if args.positional:
            self.positional_emb = torch.nn.Embedding(10,hidden_size).cuda()

    def forward(self, images, text, pos_mask):
        if self.frozen_clip:
            with torch.no_grad():
                image_features = self.clip.encode_image(images)
                text_features = self.clip.encode_text(text)
        else:
            image_features = self.clip.encode_image(images)
            text_features = self.clip.encode_text(text)
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = torch.cat(10 * [text_features])
        if self.fusion == 'concat':
            x = torch.cat((image_features, text_features), dim=1)
        else:
            x = (self.logit_scale * image_features) * text_features
        x_ = torch.unsqueeze(x,dim=0)
        if self.positional:
            embs = self.positional_emb(torch.arange(10).cuda())
            embs = embs * pos_mask
            x_pos = x_ + embs
        else:
            x_pos = x_
        attention_mask = torch.ones((self.batch_size,1,1,10)).cuda()
        x = self.transformer[0](x_pos, attention_mask)
        for layer_module in self.transformer[1:]:
            x = layer_module(x, attention_mask) #TODO: remove hard-coding of 10
        if self.add_input:
            x = x + x_
        preds = self.prediction_layer(x.half())
        return preds

    def encode_images(self, photos_batch):
        photos = [Image.open(photo_file) for photo_file in photos_batch]
        photos_preprocessed = torch.stack([self.preprocess(photo) for photo in photos]).to(device)

        with torch.no_grad():
            photos_features = self.clip.encode_image(photos_preprocessed)
            photos_features /= photos_features.norm(dim=-1, keepdim=True)
        return photos_features.cpu().numpy()

    def encode_text(self, search_query):
        with torch.no_grad():
            text_encoded = self.clip.encode_text(clip.tokenize(search_query, truncate=True).to(device))
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        return text_encoded.cpu().numpy()


config = wandb.config
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--test_descr_path', type=str, default='../../data/test_data.json')
parser.add_argument('--imgs_path', type=str, default='/network/scratch/b/benno.krojer/dataset/games')
parser.add_argument("-b", "--batchsize", type=int, default=36)
parser.add_argument("--fusion", type=str, default='mult')
parser.add_argument("-a", "--activation", default='gelu')
parser.add_argument("-s", "--logit_scale", default=1000)
parser.add_argument("--frozen_clip", default=True)
parser.add_argument("--add_input", default=True)
parser.add_argument("--positional", action="store_true")
parser.add_argument("--head_scheduler", default= 1.0, type=float)
parser.add_argument("--base_scheduler", default= 1.0, type=float)
parser.add_argument("--transformer_layers", default=2, type=int)
parser.add_argument("--job_id")

args = parser.parse_args()
assert args.fusion in ['concat', 'mult']
assert args.activation in ['leaky-relu', 'relu', 'gelu']
wandb.config.update(args)

img_dirs = args.imgs_path
valid_data = json.load(open(args.test_descr_path, 'r'))
valid = []
for img_dir, data in valid_data.items():
    for img_idx, text in data.items():
        valid.append((img_dir, img_idx, text))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'DEVICE USED: {device}')

bert_config = json.load(open('vilbert-and-bert-config.json', 'r'))
contextual_clip = ContextualCLIP(bert_config, args)
checkpoint = torch.load(args.checkpoint)
contextual_clip.load_state_dict(checkpoint['model_state_dict'])

config = wandb.config
wandb.watch(contextual_clip)
if device == "cpu":
    contextual_clip.float()
else:
    clip.model.convert_weights(
        contextual_clip)  # Actually this line is unnecessary since clip by default already on float16


correct = 0
total = 0
vid_correct = 0
vid_total = 0
img_correct= 0
img_total = 0

results = defaultdict(dict)
for img_dir, img_idx, text in tqdm.tqdm(valid):
    text = [text]
    img_idx = int(img_idx)
    img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
    img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
    images = [Image.open(photo_file) for photo_file in img_files]
    images = torch.stack([contextual_clip.preprocess(photo) for photo in images]).to(device)
    text = clip.tokenize(text, truncate=True).to(device)
    if "open-images" in str(img_dir):
        pos_mask = torch.zeros((10,1)).cuda()
    else:
        pos_mask = torch.ones((10,1)).cuda()
    with torch.no_grad():
        logits = contextual_clip(images, text, pos_mask).squeeze()
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

    total += 1
    results[img_dir].update({f'raw_preds_{img_idx}': logits.squeeze().tolist(), f'clip_pred_{img_idx}': int(pred.item()) ,f'correct_{img_idx}': 1 if img_idx == pred else 0})

print('OVERALL ACC: ' + str(round(correct/len(valid),4)))
print('VIDEO ACC: ' + str(round(vid_correct/vid_total,4)))
print('IMG ACC: ' + str(round(img_correct/img_total,4)))
json.dump(results, open(f'results/nocontra-test-data.json', 'w'), indent=2)
json.dump(results, open(f'results/CONTEXTUAL_test_set.json', 'w'), indent=2)
