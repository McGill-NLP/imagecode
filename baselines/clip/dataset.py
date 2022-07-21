import json
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Pad, Resize, ToTensor, Compose
# from transformers import BertTokenizerFast
from PIL import Image


def default_image_transform(img, img_size=224):
    img = img.convert('RGB')
    w, h = img.size
    img = Compose([
        Pad([0, (w-h)//2] if w>h else [(h-w)//2, 0]), 
        Resize([img_size, img_size]), 
        ToTensor()
    ])(img)
    return img


def default_text_transform(text, tokenizer, max_length=77):
    inputs = tokenizer(
        text,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors='np'
    )
    return inputs


class ImageCoDeDataset(Dataset):

    def __init__(self, data_dir, split, image_transform=None, text_transform=None, video_only=False):
        super().__init__()
        assert split in ['train', 'valid']

        if image_transform is not None:
            self.image_transform = image_transform
        else:
            self.image_transform = default_image_transform
        
        # if text_transform is not None:
        self.text_transform = text_transform
        # else:
        #     self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        #     self.text_transform = partial(default_text_transform, tokenizer=self.tokenizer)

        self.data = self.load_data(Path(data_dir), '/network/scratch/b/benno.krojer/dataset/games', split, video_only)

    @staticmethod
    def load_data(data_dir, img_path, split, video_only=False):
        with open(data_dir / f'{split}_data.json') as f:
            json_file = json.load(f)

        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path(f'{img_path}/{img_dir}')).glob('*.jpg'))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                static = 'open-images' in img_dir
                if video_only:
                    if not static:
                        dataset.append((img_dir, img_files, int(img_idx), text))
                else:
                    dataset.append((img_dir, img_files, int(img_idx), text))
        
        return dataset
    
    def __getitem__(self, idx):
        img_dir, img_files, img_idx, text = self.data[idx]
        
        images = [self.image_transform(Image.open(img_file)) for img_file in img_files]
        img = torch.stack(images, dim=0)
        
        txt = self.text_transform(text)
        is_video = torch.tensor(1 if 'open-images' not in img_dir else 0)
        
        return img, txt, img_idx, is_video
    
    def __len__(self):
        return len(self.data)