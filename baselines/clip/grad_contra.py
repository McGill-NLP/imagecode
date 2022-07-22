# inspired from: https://github.com/openai/CLIP/issues/83
# https://github.com/openai/CLIP/issues/83
from importlib import import_module
from itertools import count
import json
import os
import random
import wandb
import clip
from clip import model
import torch
from torch import autograd
from torch.utils.data import DataLoader
from dataset import ImageCoDeDataset
import tqdm
from torch import nn, optim
from PIL import Image
from pathlib import Path
from collections import defaultdict
import argparse
from functools import partial
random.seed(10)
torch.manual_seed(10)
wandb.init(project='clip-grad-supervision', settings=wandb.Settings(start_method='fork'))

def find_best_matches(text_features, photo_features):
    similarities = (photo_features @ text_features.T).squeeze(1)
    best_photo_idx = (-similarities).argsort()
    similarities = -similarities
    similarities.sort()
    return best_photo_idx, similarities


def convert_models_to_fp32(model):
    for name, p in model.named_parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

config = wandb.config
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=36)
parser.add_argument('--grad_accumulation', type=int, default=1)
parser.add_argument('--lr', type=float, default=4e-6)
parser.add_argument('--vit', type=str)
parser.add_argument('--decay', default=0.01, type=float)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--data_dir', type=str, default='../../data/')
parser.add_argument('--imgs_path', type=str, default='/network/scratch/b/benno.krojer/dataset/games')
parser.add_argument('--save_model', action='store_true')
parser.add_argument("--job_id")

args = parser.parse_args()
wandb.config.update(args)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'DEVICE USED: {DEVICE}')
model, preprocess = clip.load(args.vit, device=DEVICE, jit=False)
wandb.watch(model)
if DEVICE == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16


dataset_train = ImageCoDeDataset(
    data_dir=args.data_dir,
    split='train',
    image_transform=preprocess,
    text_transform=partial(clip.tokenize, truncate=True)
)
dataloader_train = DataLoader(
    dataset=dataset_train,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
dataset_valid = ImageCoDeDataset(
    data_dir=args.data_dir,
    split='valid',
    image_transform=preprocess,
    text_transform=partial(clip.tokenize, truncate=True)
)
dataloader_valid = DataLoader(
    dataset=dataset_valid,
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=args.decay)
best_val = 0

for i in range(args.epochs):
    save_model = False
    # EVALUATE
    if i > 0:
        correct = 0
        total = 0
        for image, text, target, is_video in tqdm.tqdm(dataloader_valid):
            image = image.to(DEVICE)
            text = text.to(DEVICE)
            target = target.to(DEVICE)
            is_video = is_video.to(DEVICE)

            with torch.no_grad():
                image_features = model.encode_image(image.flatten(0, 1)).view(*image.shape[:2], -1)
                text_features = model.encode_text(text.squeeze(1))
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.unsqueeze(2)).squeeze()
            prediction = similarity.argmax()

            total += 1
            correct += (prediction == target)
        acc = correct / total
        acc = acc.item()
        print(f'accuracy: {acc:.4f}')
        wandb.log({'val_acc': acc})
        if acc > best_val:
            best_val = acc
            save_model = True
            string = ''
            for key, val in list(vars(args).items()):
                if 'path' not in key:
                    string += f'_{val}'
            if args.save_model:
                torch.save({
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"checkpoints/CONTRA_clip_best_{string.replace('/', '')}.pt")

    print(f'EPOCH: {i}')
    for step, (images, text, target, is_video) in tqdm.tqdm(enumerate(dataloader_train)):
        images = images.to(DEVICE).requires_grad_()
        text = text.to(DEVICE)
        target = target.to(DEVICE)
        is_video = is_video.to(DEVICE)
        batchsize, image_size = images.shape[0], images.shape[-1]

        image_features_ = model.encode_image(images.flatten(0, 1)).reshape(*images.shape[:2], -1)
        text_features_ = model.encode_text(text.squeeze(1))
        image_features = image_features_ / image_features_.norm(dim=-1, keepdim=True)
        text_features = text_features_ / text_features_.norm(dim=-1, keepdim=True)
	
        similarity = (image_features @ text_features.unsqueeze(2)).squeeze() * model.logit_scale.exp()

        zero = torch.zeros((batchsize,1,3,image_size,image_size)).cuda()
        diff = images - torch.cat([zero, images[:,:-1,:,:,:]], dim=1)
        diff = diff - torch.cat([images[:,1:,:,:,:], zero], dim=1)

        # target_sim = similarity.gather(1, target.unsqueeze(1))
        similarity.sum().backward(retain_graph=True)
        img_grad = images.grad

        optimizer.zero_grad() #make sure that we don't update weights based on the previous backward step
        diff = diff.reshape(diff.shape[0]*diff.shape[1], diff.shape[2]*diff.shape[3]*diff.shape[4])
        img_grad = img_grad.reshape(img_grad.shape[0]*img_grad.shape[1], img_grad.shape[2]*img_grad.shape[3]*img_grad.shape[4])
        counterfactual_loss = 1 - torch.nn.functional.cosine_similarity(diff, img_grad, dim=1) #do abs() before maybe?
        counterfactual_loss = counterfactual_loss.reshape(batchsize, 10) * is_video.unsqueeze(1)
        counterfactual_loss = counterfactual_loss.mean()
        
        ground_truth = torch.tensor(target).long()  # the index of the correct one
        loss = loss_txt(similarity, ground_truth) + args.loss_factor * counterfactual_loss
        loss.backward()
        if step % args.grad_accumulation == 0:
            print(loss.item())
            wandb.log({'loss': loss})
            if DEVICE == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            optimizer.zero_grad()
