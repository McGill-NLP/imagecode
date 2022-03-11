Below are all commands for baseline experiments shown in the paper.

## CLIP
**CLIP zero-shot:**

`python3 zero_shot.py`

**CLIP (non-contrastive):**

`python3 nocontra.py`

**CLIP +ContextBatch (contrastive):**

`python3 contra.py`

For evaluation of nocontra.py and contra.py, run:

`evaluate_clip.py --checkpoint checkpoints/<MY_CHECKPOINT_FILE.pt> --test_descr_path ../../data/test_data.json`

**CLIP +ContextModule:**

`python3 contextual.py --lr 2e-6 --lr_head 1e-4 -b 36 -m ViT-B/16 --fusion mult -a gelu --logit_scale 1000 --finetuned_checkpoint_path checkpoints/CONTRA_clip_best__36_4e-06_30_1395526.pt --add_input --frozen_clip`

**CLIP +TemporalEmbedding:**

`python3 contextual.py --lr 2e-6 --lr_head 1e-4 -b 36 -m ViT-B/16 --fusion mult -a gelu --logit_scale 1000 --finetuned_checkpoint_path checkpoints/CONTRA_clip_best__36_4e-06_30_1395526.pt --add_input --frozen_clip --positional`

For evaluation for contextual.py run:

`evaluate_contextual.py --checkpoint checkpoints/<MY_CHECKPOINT_FILE.pt> --test_descr_path ../../data/test_data.json`
`evaluate_contextual.py --checkpoint checkpoints/<MY_CHECKPOINT_FILE.pt> --test_descr_path ../../data/test_data.json --positional`

## ViLBERT

**ViLBERT zero-shot**

`python3 zero_shot.py --config_file vilbert_base.json --tasks_config_file task_config/vilbert_zero_shot.yml --task 3 --output_dir results/vilbert-zero-shot --from_pretrained vilbert-pretrained.bin`

**ViLBERT (non-contrastive):**

`python nocontra.py --config_file vilbert_base.json --tasks_config_file task_config/vilbert_nocontra.yml --task 3 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 --output_dir checkpoints/vilbert --logdir logs/vilbert --from_pretrained vilbert-pretrained.bin --num_train_epochs 10 --lr 0.000005 --save_name FINAL-NOCONTRA`

**ViLBERT +ContextBatch (contrastive):**

`python3 contra.py --config_file vilbert_base.json --tasks_config_file task_config/vilbert_contra.yml --task 3 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 --output_dir checkpoints/contravilbert --logdir logs/contravilbert --from_pretrained vilbert-pretrained.bin --num_train_epochs 25 --lr 0.00004 --save_name FINAL-CONTRA --temperature 0.15 --ce_loss`

**ViLBERT +ContextModule:**

`python3 contextual.py --config_file vilbert_base.json --tasks_config_file task_config/vilbert_contextual.yml --task 3 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 --output_dir checkpoints/contravilbert --logdir logs/contravilbert --from_pretrained vilbert-pretrained.bin --num_train_epochs 25 --lr 0.00002 --save_name FINAL-CONTEXT --add_inputs --transformer_layers 4 --all_pos`

**ViLBERT +TemporalEmbedding:**

`python3 contextual.py --config_file vilbert_base.json --tasks_config_file task_config/vilbert_contextual.yml --task 3 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 --output_dir checkpoints/contravilbert --logdir logs/contravilbert --from_pretrained vilbert-pretrained.bin --num_train_epochs 25 --lr 0.00002 --save_name FINAL-TEMP-CONTEXT --add_inputs --transformer_layers 4 --positional --all_pos`

## UNITER

**UNITER zero-shot**

`python3 zero_shot.py --config_file ctrl_uniter_base.json --tasks_config_file task_config/vilbert_zero_shot.yml --task 3 --output_dir results/vilbert-zero-shot --from_pretrained uniter-pretrained.bin`

**UNITER (non-contrastive):**

`python nocontra.py --config_file ctrl_uniter_base.json --tasks_config_file task_config/vilbert_nocontra.yml --task 3 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 --output_dir checkpoints/vilbert --logdir logs/vilbert --from_pretrained uniter-pretrained.bin --num_train_epochs 10 --lr 0.000008 --save_name FINAL-NOCONTRA`

**UNITER +ContextBatch (contrastive):**

`python3 contra.py --config_file ctrl_uniter_base.json --tasks_config_file task_config/vilbert_contra.yml --task 3 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 --output_dir checkpoints/contravilbert --logdir logs/contravilbert --from_pretrained uniter-pretrained.bin --num_train_epochs 25 --lr 0.000007 --save_name FINAL-CONTRA --temperature 0.15 --ce_loss`

**UNITER +ContextModule:**

`python3 contextual.py --config_file ctrl_uniter_base.json --tasks_config_file task_config/vilbert_contextual.yml --task 3 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 --output_dir checkpoints/contravilbert --logdir logs/contravilbert --from_pretrained uniter-pretrained.bin --num_train_epochs 25 --lr 0.000006 --save_name FINAL-CONTEXT --add_inputs --transformer_layers 5 --all_pos`

**UNITER +TemporalEmbedding:**

`python3 contextual.py --config_file ctrl_uniter_base.json --tasks_config_file task_config/vilbert_contextual.yml --task 3 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 --output_dir checkpoints/contravilbert --logdir logs/contravilbert --from_pretrained uniter-pretrained.bin --num_train_epochs 25 --lr 0.000006 --save_name FINAL-TEMP-CONTEXT --add_inputs --transformer_layers 5 --positional --all_pos`

Evaluation for ViLBERT and UNITER happens already during training. This is not ideal but ensured there was no slight accuracy difference between the model during training and saved model.
