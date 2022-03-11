# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import os
import sys
import json
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
from easydict import EasyDict as edict
import json
import numpy as np

import torch
import torch.distributed as dist
# from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

from pytorch_transformers.optimization import AdamW, WarmupConstantSchedule, WarmupLinearSchedule

from volta.config import BertConfig
from volta.optimization import RAdam
from volta.encoders import BertForVLTasks
from volta.train_utils import freeze_layers, tbLogger, summary_parameters, save, resume
from volta.task_utils import LoadDataset, LoadLoss, ForwardModelsTrain, ForwardModelsVal
import wandb
wandb.init(project='finetune-contra-uniter', entity='bennokrojer', settings=wandb.Settings(start_method='fork'))
wb_conf = wandb.config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_file", default="config/vilbert_base.json", type=str,
                        help="The config file which specified the model details.")
    parser.add_argument("--resume_file", default="", type=str,
                        help="Resume from checkpoint")
    parser.add_argument("--eval_only", action='store_true')
    # Output
    parser.add_argument("--output_dir", default="save", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--logdir", default="logs", type=str,
                        help="The logging directory where the training logs will be written.")
    parser.add_argument("--save_name", default="", type=str,
                        help="save name for training.")
    # Task
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    # Training
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", dest="grad_acc_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    # Scheduler
    parser.add_argument("--lr_scheduler", default="warmup_linear", type=str,
                        help="whether use learning rate scheduler.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=None, type=float,
                        help="Number of training steps to perform linear learning rate warmup for. "
                             "It overwrites --warmup_proportion.")
    # Seed
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--in_memory", default=False, type=bool,
                        help="whether use chunck for parallel training.")
    # Optimization
    parser.add_argument("--optim", default="AdamW", type=str,
                        help="what to use for the optimization.")
    parser.add_argument("--lr", default=0.00004, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default=(0.9, 0.999), nargs="+", type=float,
                        help="Betas for Adam optimizer.")
    parser.add_argument("--adam_correct_bias", default=False, action='store_true',
                        help="Correct bias for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay for Adam optimizer.")
    parser.add_argument("--clip_grad_norm", default=0.0, type=float,
                        help="Clip gradients within the specified range.")
    parser.add_argument("--ce_loss", action="store_true")
    parser.add_argument("--job_id")

    return parser.parse_args()


def main():
    args = parse_args()

    # Devices
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}")

    # Load config
    config = BertConfig.from_json_file(args.config_file)

    # Load task config
    with open(args.tasks_config_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    task_id = args.task.strip()
    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]
    base_lr = args.lr
    print("LEARNING RATE: " + str(base_lr))
    wb_conf.lr = base_lr
    wb_conf.batch_size = task_cfg[task]["train_batch_size"]
    if task_cfg[task].get("fusion_method", None):
        # VL-BERT pooling for VQA
        config.fusion_method = task_cfg[task]["fusion_method"]

    # Output dirs
    if args.save_name:
        args.save_name = args.save_name + '_' + str(args.lr)
        prefix = "-" + args.save_name
    else:
        prefix = ""
    timestamp = (task_name + "_" + args.config_file.split(".")[0] + prefix)
    save_path = os.path.join(args.output_dir, timestamp)
    if default_gpu:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save all the hidden parameters.
        with open(os.path.join(save_path, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    if args.eval_only:
        batch_size, task2num_iters, _, dset_val, _, dl_val = LoadDataset(args, config, task_cfg, args.task, "val", contrastive_data=True)
        dset_train = dset_val
        dl_train = dl_val 
        task2num_iters = {'TASK3': len(dl_train)} 
    else:
        batch_size, task2num_iters, dset_train, dset_val, dl_train, dl_val, dset_test, dl_test = LoadDataset(args, config, task_cfg, args.task, contrastive_data=True)

    # Logging
    logdir = os.path.join(args.logdir, timestamp)
    tb_logger = tbLogger(logdir, save_path, [task_name], [task], task2num_iters, args.grad_acc_steps)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Model
    model = BertForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])
    if task_cfg[task].get("embed_clf", None):
        logger.info('Initializing classifier weight for %s from pretrained word embeddings...' % task)
        answers_word_embed = []
        for k, v in model.state_dict().items():
            if 'bert.embeddings.word_embeddings.weight' in k:
                word_embeddings = v.detach().clone()
                break
        for answer, label in sorted(dset_train.ans2label.items()):
            a_tokens = dset_train._tokenizer.tokenize(answer)
            a_ids = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)
            if len(a_ids):
                a_word_embed = (torch.stack([word_embeddings[a_id] for a_id in a_ids], dim=0)).mean(dim=0)
            else:
                a_tokens = dset_train._tokenizer.tokenize("<unk>")
                a_id = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)[0]
                a_word_embed = word_embeddings[a_id]
            answers_word_embed.append(a_word_embed)
        answers_word_embed_tensor = torch.stack(answers_word_embed, dim=0)
        for name, module in model.named_modules():
            if name.endswith('clfs_dict.%s.logit_fc.3' % task):
                module.weight.data = answers_word_embed_tensor.to(device=module.weight.data.device)
    print("LOADED MODEL")
    # Optimization details
    
    ce = torch.nn.CrossEntropyLoss()
    def CE_loss(output, target):
        total = []
        output = torch.squeeze(output)
        target = torch.squeeze(target)
        for i in range(0,target.shape[0], 10):
            sub_out = output[i:i+10]
            sub_target = target[i:i+10]
            pos_index = (sub_target == 1).nonzero(as_tuple=True)[0]
            loss = ce(sub_out.unsqueeze(dim=0), pos_index)
            total.append(loss)
        total = torch.stack(total).mean()
        return total

    if True:
        print("OPTIMIZATION DETAILS")
        freeze_layers(model)
        criterion = CE_loss
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if "vil_" in key:
                    lr = 1e-4
                else:
                    lr = base_lr
                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": 0.0}]
                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": args.weight_decay}]
        if default_gpu:
            print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))
        if args.optim == "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=base_lr,
                              eps=args.adam_epsilon,
                              betas=args.adam_betas,
                              correct_bias=args.adam_correct_bias)
        elif args.optim == "RAdam":
            optimizer = RAdam(optimizer_grouped_parameters, lr=base_lr)
        num_train_optim_steps = (task2num_iters[task] * args.num_train_epochs // args.grad_acc_steps)
        warmup_steps = args.warmup_steps or args.warmup_proportion * num_train_optim_steps
        if args.lr_scheduler == "warmup_linear":
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optim_steps)
        else:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)
    
        # Resume training
        print("RESUMING TRAINING")
        start_iter_id, global_step, start_epoch, tb_logger, max_score = \
            resume(args.resume_file, model, optimizer, scheduler, tb_logger)
        print("LOADED RESUMED TRAINING")
        # Move to GPU(s)
        model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
                )
            model = DDP(model, delay_allreduce=True)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)
    
        # Save starting model
        #if start_epoch == 0:
            #save(save_path, logger, -1, model, optimizer, scheduler, global_step, tb_logger, default_gpu, max_score) 
        # Print summary
        if default_gpu:
            summary_parameters(model, logger)
            print("***** Running training *****")
            print("  Num Iters: ", task2num_iters[task])
            print("  Batch size: ", batch_size)
            print("  Num steps: %d" % num_train_optim_steps)
    
    # Train
    if args.eval_only:
        print("EVAL")
        model.to(device)
        model.eval()
        score = evaluate(config, dl_val, task_cfg, device, task, criterion, model, -1, default_gpu, tb_logger, args)
    else:
        for epoch_id in tqdm(range(start_epoch, args.num_train_epochs), desc="Epoch"):
            model.train()
            for step, batch in tqdm(enumerate(dl_train), desc="Batch"):
                batch[7] = torch.stack(batch[7], dim=0)
                batch[7] = torch.transpose(batch[7], 0, 1)
                for i in range(len(batch)):
                    s = list(batch[i].shape)
                    s = [s[0]*s[1]] + s[2:]
                    batch[i] = batch[i].reshape(s)

                iter_id = start_iter_id + step + (epoch_id * len(dl_train))
    
                loss = ForwardModelsTrain(config, task_cfg, device, task, batch, model, criterion)
    
                if args.grad_acc_steps > 1:
                    loss = loss / args.grad_acc_steps
                loss.backward()
    
                if (step + 1) % args.grad_acc_steps == 0:
                    # Clip gradient
                    if args.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    
                    optimizer.step()
    
                    if global_step < warmup_steps or args.lr_scheduler == "warmup_linear":
                        scheduler.step()
    
                    model.zero_grad()
                    global_step += 1
    
                    if default_gpu:
                        wandb.log({'loss': loss.item()})
    
                if (step % (20 * args.grad_acc_steps) == 0) and step != 0 and default_gpu:
                    tb_logger.showLossTrain()
    
                # Decide whether to evaluate task
                if iter_id != 0 and iter_id % task2num_iters[task] == 0:
#                if True:
                    score = evaluate(config, dl_val, task_cfg, device, task, criterion, model, epoch_id, default_gpu, tb_logger, args)
                    score = evaluate(config, dl_test, task_cfg, device, task, criterion, model, epoch_id, default_gpu, tb_logger, args, val=False)
                    if score > max_score:
                        max_score = score
                        #save(save_path, logger, epoch_id, model, optimizer, scheduler,
                        #     global_step, tb_logger, default_gpu, max_score, is_best=True)
    
            #save(save_path, logger, epoch_id, model, optimizer, scheduler, global_step, tb_logger, default_gpu, max_score)
        
    
    tb_logger.txt_close()


def evaluate(config, dataloader_val, task_cfg, device, task_id, criterion, model, epoch_id, default_gpu, tb_logger, args, val=True):
    model.eval()
    question_ids = []
    preds = []
    for i, batch in enumerate(dataloader_val):
        batch[7] = torch.stack(batch[7], dim=0)
        batch[7] = torch.transpose(batch[7], 0, 1)
        for i in range(len(batch)):
            s = list(batch[i].shape)
            s = [s[0]*s[1]] + s[2:]
            batch[i] = batch[i].reshape(s)

        loss, batch_size, vil_prediction, question_id = ForwardModelsVal(config, task_cfg, device, task_id, batch, model, criterion)
        question_ids += question_id.tolist()
        preds += vil_prediction.tolist()
        if default_gpu:
            sys.stdout.write("%d/%d\r" % (i, len(dataloader_val)))
            sys.stdout.flush()

    groups = defaultdict(list)
    for pred, qid in zip(preds, question_ids):
        img_set = str(qid)[:-2]
        groups[img_set].append([str(qid), pred])
    total = 0
    total_corr = 0
    for _, preds in groups.items():
        preds = sorted(preds, key = lambda x: x[1], reverse=True)
        rank = -1
        for i, p in enumerate(preds):
            key = p[0]
            if key[-1] == key [-2]:
                rank = i+1
        if rank == 1:
            total_corr += 1
        total += 1
    score = total_corr / total
    if val:
        print(f"SCORE ON VAL SET: {score}")
        json.dump(groups, open(f'results/VAL_{args.save_name}_epoch{epoch_id}_score{round(score,3)}.json', 'w'))
        wandb.log({'val_acc': score})
        model.train()
    else:
        print(f"SCORE ON TEST SET: {score}")
        json.dump(groups, open(f'results/TEST_{args.save_name}_epoch{epoch_id}_score{round(score,3)}.json', 'w'))
        wandb.log({'test_acc': score})
        model.train()
 
    return score

if __name__ == "__main__":
    main()
