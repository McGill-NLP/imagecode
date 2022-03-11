# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import _pickle as cPickle
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(question, answer):
    answer.pop("image_id")
    answer.pop("question_id")
    entry = {
        "question_id": question["question_id"],
        "image_id": question["image_id"],
        "question": question["question"],
        "answer": answer,
    }
    return entry


def _load_dataset(dataroot, name):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'
    """
    question_path = os.path.join(dataroot, f"{name}.json")
    questions = sorted(json.load(open(question_path))["questions"], key=lambda x: x["question_id"])
    answer_path = os.path.join(dataroot, "%s_target.pkl" % name)
    answers = cPickle.load(open(answer_path, "rb"))
    answers = sorted(answers, key=lambda x: x["question_id"])



    assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        assert_eq(question["question_id"], answer["question_id"])
        assert_eq(question["image_id"], answer["image_id"])
        entries.append(_create_entry(question, answer))

    return entries


class VQAClassificationDataset(Dataset):
    def __init__(
        self,
        task,
        dataroot,
        annotations_jsonpath,
        split,
        image_features_reader,
        gt_image_features_reader,
        tokenizer,
        bert_model,
        padding_index=0,
        max_seq_length=16,
        max_region_num=101,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        super().__init__()
        self.split = split
        ans2label_path = os.path.join(dataroot, "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataroot, "trainval_label2ans.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat
        self._append_mask_sep = append_mask_sep

        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
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
                task
                + "_"
                + split
                + "_"
                + str(max_seq_length) +
                ".pkl",
            )
        if not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split)
            self.tokenize(max_seq_length)
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

        self.qid2imgid = {e["question_id"]: e["image_id"] for e in self.entries}

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["question"])
            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]

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
        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
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

        if self._append_mask_sep:
            mask_id = self._tokenizer.convert_tokens_to_ids(['[MASK]', '[CLS]'])
            mask_pos = input_mask.sum()
            question = torch.cat([question[:mask_pos], torch.tensor(mask_id), question[mask_pos:]])
            input_mask = torch.cat([input_mask[:mask_pos], torch.tensor([1, 1]), input_mask[mask_pos:]])
            segment_ids = torch.cat([segment_ids[:mask_pos], torch.tensor([1, 1]), segment_ids[mask_pos:]])

        target = torch.zeros(self.num_labels)

        if "test" not in self.split:
            answer = entry["answer"]
            labels = answer["labels"]
            scores = answer["scores"]
            if labels is not None:
                target.scatter_(0, labels, scores)

        return features, spatials, image_mask, question, target, input_mask, segment_ids, question_id

    def __len__(self):
        return len(self.entries)

class ContrastiveVQAClassificationDataset(Dataset):
    def __init__(
        self,
        task,
        dataroot,
        annotations_jsonpath,
        split,
        image_features_reader,
        gt_image_features_reader,
        tokenizer,
        bert_model,
        padding_index=0,
        max_seq_length=16,
        max_region_num=101,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        super().__init__()
        self.split = split
        ans2label_path = os.path.join(dataroot, "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataroot, "trainval_label2ans.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat
        self._append_mask_sep = append_mask_sep

        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
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
                task
                + "_"
                + split
                + "_"
                + str(max_seq_length) +
                ".pkl",
            )
        if not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split)
            self.tokenize(max_seq_length)
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

        self.qid2imgid = {e["question_id"]: e["image_id"] for e in self.entries}
        new_entries = []
        for i in range(0,len(self.entries), 10):
            new_entry = {}
            for k in self.entries[i]:
                if k == 'answer':
                    new_labels = [torch.tensor(j[k]['labels']) for j in self.entries[i:i+10]]
                    new_scores = [torch.tensor(j[k]['scores'], dtype=torch.float) for j in self.entries[i:i+10]]
                    # print(f'\n\nNEWLABELS: {new_labels}\n\nEND')
                    new_labels = torch.stack(new_labels, dim=0)
                    new_scores = torch.stack(new_scores, dim=0)
                    new_entry[k] = {'labels': new_labels, 'scores': new_scores}
                else:
                    new_v = [j[k] for j in self.entries[i:i+10]]
                    if torch.is_tensor(new_v[0]):
                        new_v = torch.stack(new_v, dim=0)
                    new_entry[k] = new_v
            new_entries.append(new_entry)
        self.entries = new_entries


    def tokenize(self, max_length=16):
        """Tokenizes the questions.
        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["question"])
            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]

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

            if True: #used to be: if "test" not in self.split
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
        image_ids = entry["image_id"]
        question_id = entry["question_id"]
        features_l = []
        image_mask_l = []
        spatials_l = []
        question_l = []
        target_l = []
        input_mask_l = []
        segment_ids_l = []
        for i in range(len(image_ids)):
            features, num_boxes, boxes, _ = self._image_features_reader[image_ids[i]]

            mix_num_boxes = min(int(num_boxes), self._max_region_num)
            mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
            mix_features_pad = np.zeros((self._max_region_num, 2048))

            image_mask = [1] * (int(mix_num_boxes))
            while len(image_mask) < self._max_region_num:
                image_mask.append(0)

            mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
            mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

            features = torch.tensor(mix_features_pad).float()
            image_mask = torch.tensor(image_mask).long()
            spatials = torch.tensor(mix_boxes_pad).float()
            features_l.append(features)
            image_mask_l.append(image_mask)
            spatials_l.append(spatials)

            question = entry["q_token"][i]
            input_mask = entry["q_input_mask"][i]
            segment_ids = entry["q_segment_ids"][i]

            if self._append_mask_sep:
                mask_id = self._tokenizer.convert_tokens_to_ids(['[MASK]', '[CLS]'])
                mask_pos = input_mask.sum()
                question = torch.cat([question[:mask_pos], torch.tensor(mask_id), question[mask_pos:]])
                input_mask = torch.cat([input_mask[:mask_pos], torch.tensor([1, 1]), input_mask[mask_pos:]])
                segment_ids = torch.cat([segment_ids[:mask_pos], torch.tensor([1, 1]), segment_ids[mask_pos:]])

            target = torch.zeros(self.num_labels)

            if True: #used to be: if "test" not in self.split
                answer = entry["answer"]
                labels = answer["labels"][i]
                scores = answer["scores"][i]
                if labels is not None:
                    target.scatter_(0, labels, scores)
            question_l.append(question)
            target_l.append(target)
            input_mask_l.append(input_mask)
            segment_ids_l.append(segment_ids)
        
        features = torch.stack(features_l, dim=0)
        image_mask = torch.stack(image_mask_l, dim=0)
        spatials = torch.stack(spatials_l, dim=0)
        question = torch.stack(question_l, dim=0)
        target = torch.stack(target_l, dim=0)
        input_mask = torch.stack(input_mask_l, dim=0)
        segment_ids = torch.stack(segment_ids_l, dim=0)

        return features, spatials, image_mask, question, target, input_mask, segment_ids, question_id

    def __len__(self):
        return len(self.entries)
