# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import jsonlines
import _pickle as cPickle
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ._image_features_reader import ImageFeaturesH5Reader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

LABEL_MAP = {"Yes": 0, "No": 1, "N/A": 2}


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(item):
    entry = {
        "question_id": item["question_id"],
        "image_id": item["image_id"],
        "question": item["question"],
        "answer": item,
    }
    return entry


def _load_dataset(dataroot, name):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'valid', 'test'
    """
    if name == "train" or name == "valid" or name == "test":
        annotations_path = os.path.join(dataroot, "guesswhat.%s.jsonl" % name)
        with jsonlines.open(annotations_path) as reader:
            # Build an index which maps image id with a list of qa annotations.
            items = []
            for annotation in reader:
                for q in annotation["qas"]:
                    dictionary = {}
                    dictionary["image_id"] = annotation["image"]["id"]
                    dictionary["question_id"] = q["id"]
                    dictionary["question"] = q["question"]
                    dictionary["labels"] = [int(LABEL_MAP[str(q["answer"])])]
                    dictionary["scores"] = [1.0]
                    items.append(dictionary)
    else:
        assert False, "data split is not recognized."

    if "test" in name:
        entries = []
        for item in items:
            entries.append(item)
    else:
        entries = []
        for item in items:
            entries.append(_create_entry(item))
    return entries


class GuessWhatDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: AutoTokenizer,
        padding_index: int = 0,
        max_seq_length: int = 16,
        max_region_num: int = 37,
    ):
        super().__init__()
        self.split = split
        self.num_labels = 3
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        cache_path = os.path.join(
            dataroot, "cache", task + "_" + split + "_" + str(max_seq_length) + ".pkl"
        )
        if not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split)
            self.tokenize(max_seq_length)
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            # tokens = self._tokenizer.tokenize(entry["question"])
            # tokens = ["[CLS]"] + tokens + ["[SEP]"]

            # tokens = [
            #     self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
            #     for w in tokens
            # ]
            tokens = self._tokenizer.encode(entry["question"])
            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]

            # tokens = tokens[:max_length]
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tensorize(self):

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            if "test" not in self.split:
                answer = entry["answer"]
                labels = np.array(answer["labels"])
                scores = np.array(answer["scores"], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry["answer"]["labels"] = labels
                    entry["answer"]["scores"] = scores
                else:
                    entry["answer"]["labels"] = None
                    entry["answer"]["scores"] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]
        question_id = entry["question_id"]
        features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
        target = torch.zeros(self.num_labels)

        if "test" not in self.split:
            answer = entry["answer"]
            labels = answer["labels"]
            scores = answer["scores"]
            if labels is not None:
                target.scatter_(0, labels, scores)

        return (
            features,
            spatials,
            image_mask,
            question,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            question_id,
        )

    def __len__(self):
        return len(self.entries)
