# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer

from volta.datasets import DatasetMapTrain, DatasetMapEval
from volta.datasets._image_features_reader import ImageFeaturesH5Reader


logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
}


def ForwardModelsVal(config, task_cfg, device, task_id, batch, model, criterion, id2name=None, all_pos=None):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_cfg[task_id]["type"] == "V-logit-mc":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multi_choice_ids, question_id = batch
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_id = batch
    if id2name is not None:
        if not all_pos:
            mask = []
            for i in range(question_id.shape[0]):
                mask.append(not 'openimages' in id2name[str(question_id[i].item())])
            pos_mask = torch.tensor(mask).cuda()
        else:
            pos_mask = torch.ones(features.shape[0]).cuda()
    else:
        pos_mask = None
    batch_size = features.size(0)
    if task_cfg[task_id]["process"] in ["dialog"]:
        raise NotImplementedError("dialog process for validation")

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

    vil_prediction, vision_prediction, linguisic_prediction, _ = model(question, features, spatials, task_id, token_type_ids=segment_ids, attention_mask=input_mask, image_attention_mask=image_mask, pos_mask=pos_mask)
    
    target = target[:,1:]
    if task_cfg[task_id]["type"] == "VL-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_prediction.view(batch_size, num_options)
        loss = criterion(vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = (preds == target).sum()

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vil_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = criterion(vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = (preds == target).sum()

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum()

    return float(loss), batch_size, vil_prediction, question_id


def ForwardModelsTrain(config, task_cfg, device, task_id, batch, model, criterion, id2name=None, all_pos=None):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_cfg[task_id]["type"] == "V-logit-mc":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multi_choice_ids, question_id = batch
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_id = batch
    if id2name is not None:
        if not all_pos:
            mask = []
            for i in range(question_id.shape[0]):
                mask.append(not 'openimages' in id2name[str(question_id[i].item())])
            pos_mask = torch.tensor(mask).cuda()
        else:
            pos_mask = torch.ones(features.shape[0]).cuda()
    else:
        pos_mask = None
    batch_size = features.size(0)
    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(rbatch_size, input_mask.size(2), input_mask.size(3))
        segment_ids = segment_ids.view(rbatch_size, segment_ids.size(2), segment_ids.size(3))

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

    vil_prediction, vision_prediction, linguisic_prediction, _ = model(question, features, spatials, task_id,
                                                                       segment_ids, input_mask, image_mask, pos_mask)

    # for different task, we use different output to calculate the loss.
    target = target[:,1:]
    if task_cfg[task_id]["type"] == "VL-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_prediction.view(batch_size, num_options)
        loss = criterion(vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = criterion(vil_prediction, target)

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = criterion(vil_prediction, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vil_prediction, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = float(torch.sum(select_target > 0.5)) / batch_size

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = criterion(vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = criterion(vil_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)

    return loss


def LoadLoss(task_cfg, task_id):
    task = "TASK" + task_id
    loss = LossMap[task_cfg[task]["loss"]]
    return loss


def LoadDataset(args, config, task_cfg, task_id, split="trainval", contrastive_data=False):
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)

    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]

    # initialize the feature reader
    feats_h5path1 = task_cfg[task]["features_h5path1"]
    feats_h5path2 = task_cfg[task]["features_h5path2"]
    features_reader1 = ImageFeaturesH5Reader(feats_h5path1, config, args.in_memory) if feats_h5path1 != "" else None
    features_reader2 = ImageFeaturesH5Reader(feats_h5path2, config, args.in_memory) if feats_h5path2 != "" else None

    train_batch_size = task_cfg[task]["train_batch_size"] // args.grad_acc_steps
    val_batch_size = task_cfg[task]["val_batch_size"] // args.grad_acc_steps
    num_workers = args.num_workers
    if args.local_rank != -1:
        train_batch_size = int(train_batch_size / dist.get_world_size())
        val_batch_size = int(val_batch_size / dist.get_world_size())
        num_workers = int(num_workers / dist.get_world_size())

    logger.info("Loading %s Dataset with batch size %d" % (task_name, train_batch_size))
    dset_train, dl_train, task2num_iters = None, None, {}
    if "train" in split:
        postfix = 'contra' if contrastive_data else ''
        dset_train = DatasetMapTrain[task_name+postfix](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
            split=task_cfg[task]["train_split"],
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
        )
        if args.local_rank == -1:
            train_sampler = RandomSampler(dset_train)
        else:
            train_sampler = DistributedSampler(dset_train)
        dl_train = DataLoader(
            dset_train,
            sampler=train_sampler,
            batch_size=train_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=args.drop_last,
        )
        task2num_iters = {task: len(dl_train)}

    dset_val, dl_val, dset_test, dl_test = None, None, None, None
    if "val" in split:
        postfix = 'contra' if contrastive_data else ''
        dset_val = DatasetMapTrain[task_name+postfix](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=task_cfg[task]["val_split"],
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
        )
        dl_val = DataLoader(
            dset_val,
            shuffle=False,
            batch_size=val_batch_size,
            num_workers=2,
            pin_memory=True,
            drop_last=args.drop_last,
        )
        dset_test = DatasetMapTrain[task_name+postfix](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split="test",
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
        )
        dl_test = DataLoader(
            dset_test,
            shuffle=False,
            batch_size=val_batch_size,
            num_workers=2,
            pin_memory=True,
            drop_last=args.drop_last,
        )


    return train_batch_size, task2num_iters, dset_train, dset_val, dl_train, dl_val, dset_test, dl_test


def LoadDatasetEval(args, config, task_cfg, task_id):
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)

    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]

    # initialize the feature reader
    feats_h5path1 = task_cfg[task]["features_h5path1"]
    feats_h5path2 = task_cfg[task]["features_h5path2"]
    print(feats_h5path1)
    features_reader1 = ImageFeaturesH5Reader(feats_h5path1, config, args.in_memory) if feats_h5path1 != "" else None
    features_reader2 = ImageFeaturesH5Reader(feats_h5path2, config, args.in_memory) if feats_h5path2 != "" else None

    batch_size = task_cfg[task].get("val_batch_size", 32) #TODO: not hardcode
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())

    logger.info("Loading %s Dataset with batch size %d" % (task_name, batch_size))
    eval_split = task_cfg[task]["val_split"]

    if task_name.startswith("Retrieval"):
        dset_val = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
            num_subiters=args.num_subiters,
        )
    else:
        dset_val = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
            image_features_reader=features_reader1,
            gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
            num_locs=config.num_locs,
            add_global_imgfeat=config.add_global_imgfeat,
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
        )

    dl_val = DataLoader(
        dset_val,
        shuffle=False,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
        drop_last=args.drop_last,
    )
    task2num_iters = {task: len(dl_val)}

    return batch_size, task2num_iters, dset_val, dl_val


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


def EvaluatingModel(config, task_cfg, device, task_id, batch, model, dataloader, criterion, results, others):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    if task_cfg[task_id]["type"] == "V-logit-mc":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multi_choice_ids, question_id = batch
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_id = batch

    batch_size = features.size(0)

    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(
            rbatch_size, input_mask.size(2), input_mask.size(3)
        )
        segment_ids = segment_ids.view(
            rbatch_size, segment_ids.size(2), segment_ids.size(3)
        )

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.v_feature_size)
            .contiguous()
            .view(-1, max_num_bbox, config.v_feature_size)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, config.num_locs)
            .contiguous()
            .view(-1, max_num_bbox, config.num_locs)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(batch_size * 2, int(features.size(1) / 2), features.size(2))
        spatials = spatials.view(batch_size * 2, int(spatials.size(1) / 2), spatials.size(2))
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))

    with torch.no_grad():
#            vil_prediction, vision_prediction, linguisic_prediction, _ = model(question, features, spatials,token_type_ids=segment_ids, attention_mask=input_mask, image_attention_mask=image_mask, output_all_encoded_layers=True)
            prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, all_attention_mask, pooled_output, encoded_layers_t, encoded_layers_v = model(question, features, spatials,token_type_ids=segment_ids, attention_mask=input_mask, image_attention_mask=image_mask, output_all_encoded_layers=True)
    for i in range(seq_relationship_score.shape[0]):
        results.append(
                {
                    "question_id": question_id[i].item(),
                    "prediction_aligned": seq_relationship_score[i][0].item(),
                    "prediction_notaligned": seq_relationship_score[i][1].item(),
                    "target_aligned": target[i][0].item()
                }
        )
    seq_relationship_prob = torch.softmax(seq_relationship_score, dim=1)
    loss = criterion(seq_relationship_prob, target)
    batch_score = 0
#    if task_cfg[task_id]["type"] == "VL-classifier": 
#        logits = torch.max(vil_prediction, 1)[1].data  # argmax
#        loss = 0
#        batch_score = 0
#        for i in range(logits.size(0)):
#            results.append(
#                {
#                    "question_id": question_id[i].item(),
#                    "answer": dataloader.dataset.label2ans[logits[i].item()],
#                }
#            )
#
#    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
#        logits = torch.max(vil_prediction, 1)[1].data
#        loss = 0
#        batch_score = 0
#        for i in range(logits.size(0)):
#            results.append(
#                {
#                    "questionId": str(question_id[i].item()),
#                    "prediction": dataloader.dataset.label2ans[logits[i].item()],
#                }
#            )
#
#    elif task_cfg[task_id]["type"] == "VL-logit":
#        vil_logit = vil_prediction.view(batch_size, num_options)
#        loss = criterion(vil_logit, target)
#        _, preds = torch.max(vil_logit, 1)
#        batch_score = (preds == target).sum()
#
#        probs = torch.softmax(vil_logit, dim=1)
#        for i in range(vil_logit.size(0)):
#            results.append(
#                {
#                    "question_id": question_id[i].item(),
#                    "answer": [prob.item() for prob in probs[i]],
#                }
#            )
#
#    elif task_cfg[task_id]["type"] == "V-logit":
#        loss = criterion(vil_prediction, target)
#        loss = loss.mean() * target.size(1)
#        _, select_idx = torch.max(vil_prediction, dim=1)
#        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
#        batch_score = torch.sum(select_target > 0.5).item()
#
#        for i in range(select_idx.size(0)):
#            results.append(
#                {
#                    "id": question_id[i].item(),
#                    "target": select_idx[i].item(),
#                    "IOU": select_target[i].item(),
#                }
#            )
#
#    elif task_cfg[task_id]["type"] == "V-logit-mc":
#        vision_logit = vil_prediction[:, 101:]  # FIXME from ViLBERT
#        vision_logit = vision_logit.squeeze(2).gather(1, multi_choice_ids)
#        vision_logit = vision_logit.unsqueeze(2)
#        loss = criterion(vision_logit, target)
#        loss = loss.mean() * target.size(1)
#        _, preds = torch.max(vision_logit, dim=1)
#        _, target = torch.max(target, dim=1)
#        batch_score = float((preds == target).sum())
#
#        for i in range(preds.size(0)):
#            results.append({"id": question_id[i].item(), "target": preds[i].item()})
#
#    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
#        loss = criterion(vil_prediction, target)
#        loss = loss.mean()
#        batch_score = compute_score_with_logits(vil_prediction, target).sum()
#
#    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
#        loss = criterion(vil_prediction, target)
#        loss = loss.mean(
    return float(loss), float(batch_score), batch_size, results, others

