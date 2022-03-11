# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import random
import jsonlines
import _pickle as cPickle

import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from ._image_features_reader import ImageFeaturesH5Reader


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _load_annotations(annotations_jsonpath, task):
    with jsonlines.open(annotations_jsonpath) as reader:
        # Build an index which maps image id with a list of caption annotations.
        entries = []
        imgid2entry = {}
        count = 0
        for annotation in reader:
            if task == "RetrievalCOCO":
                image_id = annotation["id"]
            elif task == "RetrievalFlickr30k":
                image_id = int(annotation["img_path"].split(".")[0])
            imgid2entry[image_id] = []
            for sentences in annotation["sentences"]:
                entries.append({"caption": sentences, "image_id": image_id})
                imgid2entry[image_id].append(count)
                count += 1
    return entries, imgid2entry


class RetrievalDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: AutoTokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 36,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        self._entries, self.imgid2entry = _load_annotations(annotations_jsonpath, task)
        self.image_id_list = [*self.imgid2entry]

        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self.num_labels = 1
        self._split = split
        self._padding_index = padding_index
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat

        if self._split == "train":
            image_info = cPickle.load(open(os.path.join(dataroot, "hard_negative" + ".pkl"), "rb"))
            for key, value in image_info.items():
                setattr(self, key, value)
            self.train_imgId2pool = {imageId: i for i, imageId in enumerate(self.train_image_list)}

        os.makedirs(os.path.join(dataroot, "annotations", "cache"), exist_ok=True)
        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "annotations",
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "annotations",
                "cache",
                task
                + "_"
                + split
                + "_"
                + str(max_seq_length)
                + ".pkl",
            )

        if not os.path.exists(cache_path):
            self.tokenize()
            self.tensorize()
            cPickle.dump(self._entries, open(cache_path, "wb"))
        else:
            print("loading entries from %s" % cache_path)
            self._entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._entries:

            tokens = self._tokenizer.encode(entry["caption"])
            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self._entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):
        entry = self._entries[index]
        image_id = entry["image_id"]

        features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features1 = torch.tensor(mix_features_pad).float()
        image_mask1 = torch.tensor(image_mask).long()
        spatials1 = torch.tensor(mix_boxes_pad).float()

        caption1 = entry["token"]
        input_mask1 = entry["input_mask"]
        segment_ids1 = entry["segment_ids"]
        # negative samples.
        # 1: correct one, 2: random caption wrong, 3: random image wrong. 4: hard image wrong.

        while True:
            # sample a random image:
            img_id2 = random.choice(self.image_id_list)
            if img_id2 != image_id:
                break

        entry2 = self._entries[random.choice(self.imgid2entry[img_id2])]

        features2 = features1
        image_mask2 = image_mask1
        spatials2 = spatials1
        caption2 = entry2["token"]
        input_mask2 = entry2["input_mask"]
        segment_ids2 = entry2["segment_ids"]

        # random image wrong
        while True:
            # sample a random image:
            img_id3 = random.choice(self.image_id_list)
            if img_id3 != image_id:
                break

        features3, num_boxes3, boxes3, _ = self._image_features_reader[img_id3]
        image_mask3 = [1] * (int(num_boxes3))

        mix_num_boxes3 = min(int(num_boxes3), self._max_region_num)

        while len(image_mask3) < self._max_region_num:
            image_mask3.append(0)

        mix_boxes_pad[:mix_num_boxes3] = boxes3[:mix_num_boxes3]
        mix_features_pad[:mix_num_boxes3] = features3[:mix_num_boxes3]

        features3 = torch.tensor(mix_features_pad).float()
        image_mask3 = torch.tensor(image_mask3).long()
        spatials3 = torch.tensor(mix_boxes_pad).float()

        caption3 = caption1
        input_mask3 = input_mask1
        segment_ids3 = segment_ids1

        if self._split == "train":
            # random hard caption.
            rand_img_id_pool = self.train_hard_pool[self.train_imgId2pool[image_id]]
            pool_img_idx = int(rand_img_id_pool[np.random.randint(1, len(rand_img_id_pool))])
            img_id4 = self.train_image_list[pool_img_idx]
        else:
            while True:
                # sample a random image:
                img_id4 = random.choice(self.image_id_list)
                if img_id4 != image_id:
                    break

        entry4 = self._entries[random.choice(self.imgid2entry[img_id4])]

        features4 = features1
        image_mask4 = image_mask1
        spatials4 = spatials1
        caption4 = entry4["token"]
        input_mask4 = entry4["input_mask"]
        segment_ids4 = entry4["segment_ids"]

        features = torch.stack([features1, features2, features3, features4], dim=0)
        spatials = torch.stack([spatials1, spatials2, spatials3, spatials4], dim=0)
        image_mask = torch.stack([image_mask1, image_mask2, image_mask3, image_mask4], dim=0)
        caption = torch.stack([caption1, caption2, caption3, caption4], dim=0)
        input_mask = torch.stack([input_mask1, input_mask2, input_mask3, input_mask4], dim=0)
        segment_ids = torch.stack([segment_ids1, segment_ids2, segment_ids3, segment_ids4], dim=0)
        target = 0

        return features, spatials, image_mask, caption, target, input_mask, segment_ids, image_id

    def __len__(self):
        return len(self._entries)


def _load_annotationsVal(annotations_jsonpath, task):
    with jsonlines.open(annotations_jsonpath) as reader:
        # Build an index which maps image id with a list of caption annotations.
        image_entries = {}
        caption_entries = []
        for annotation in reader:
            if task == "RetrievalCOCO":
                image_id = annotation["id"]
            elif task == "RetrievalFlickr30k":
                image_id = int(annotation["img_path"].split(".")[0])
            image_entries[image_id] = 1
            for sentences in annotation["sentences"]:
                caption_entries.append({"caption": sentences, "image_id": image_id})
    image_entries = ['img'+str(i) for i in range(10)]
    return image_entries, caption_entries


class RetrievalDatasetVal(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: AutoTokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 36,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
        num_subiters=2,
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        self._image_entries, self._caption_entries = _load_annotationsVal(annotations_jsonpath, task)
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer

        self._split = split
        self._padding_index = padding_index
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat
        self.num_labels = 1

        self.tokenize()
        self.tensorize()

        self.num_subiters = num_subiters
        self.num_images = len(self._image_entries)
        self.max_num_images = self.num_images // self.num_subiters
        print(self.num_images)
        print(self.max_num_images)
        self.batches_per_caption = self.num_images // self.max_num_images + self.num_images % self.max_num_images

        self.features_all = np.zeros((len(self._image_entries), self._max_region_num, 2048))
        self.spatials_all = np.zeros((len(self._image_entries), self._max_region_num, self._num_locs))
        self.image_mask_all = np.zeros((len(self._image_entries), self._max_region_num))
        success = False
        for i, image_id in enumerate(self._image_entries):
            features, num_boxes, boxes, _ = self._image_features_reader[image_id]

            mix_num_boxes = min(int(num_boxes), self._max_region_num)
            mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
            mix_features_pad = np.zeros((self._max_region_num, 2048))
    
            image_mask = [1] * (int(mix_num_boxes))
            while len(image_mask) < self._max_region_num:
                image_mask.append(0)

            mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
            mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

            self.features_all[i] = mix_features_pad
            self.image_mask_all[i] = np.array(image_mask)
            self.spatials_all[i] = mix_boxes_pad

            sys.stdout.write("%d/%d\r" % (i, len(self._image_entries)))
            sys.stdout.flush()
        self.features_all = torch.Tensor(self.features_all).float()
        self.image_mask_all = torch.Tensor(self.image_mask_all).long()
        self.spatials_all = torch.Tensor(self.spatials_all).float()

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._caption_entries:
            tokens = self._tokenizer.encode(entry["caption"])
            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self._caption_entries:
            token = torch.from_numpy(np.array(entry["token"])).long()
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"])).long()
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):
        # we iterate through every caption here.
        caption_idx = int(index / self.num_subiters)
        image_idx = index % self.num_subiters

        image_entries = self._image_entries[self.max_num_images * (image_idx):self.max_num_images * (image_idx + 1)]
        features_all = self.features_all[self.max_num_images * (image_idx):self.max_num_images * (image_idx + 1)]
        spatials_all = self.spatials_all[self.max_num_images * (image_idx):self.max_num_images * (image_idx + 1)]
        image_mask_all = self.image_mask_all[self.max_num_images * (image_idx):self.max_num_images * (image_idx + 1)]

        entry = self._caption_entries[caption_idx]
        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]

        target_all = torch.zeros(self.max_num_images)
        for i, image_id in enumerate(image_entries):
            if image_id == entry["image_id"]:
                target_all[i] = 1

        return (
            features_all,
            spatials_all,
            image_mask_all,
            caption,
            input_mask,
            segment_ids,
            target_all,
            caption_idx,
            image_idx,
        )

    def __len__(self):
        return len(self._caption_entries) * self.num_subiters
