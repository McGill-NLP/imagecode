import json
import math
from collections import defaultdict
import torch
from torch.nn.functional import softmax
import sys

filename = sys.argv[1]
split = sys.argv[2]
data = json.load(open(filename, 'r'))
mapping = json.load(open(f'counter2key_{split}.json', 'r'))
ids = json.load(open('shortid2id.json', 'r'))

results = defaultdict(dict)
for img_set, v in data.items():
    real_id = ids[mapping[v[0][0]]]
    real_id = real_id.split('___')[0]
    target = int(str(v[0][0])[-2])

    preds = []
    pred_idx = 0
    best_pred = -math.inf
    for j in range(10):
        pred = v[j][1][0]
        preds.append(pred)
        if pred > best_pred:
            best_pred = pred
            pred_idx = j

    results[real_id].update({f'raw_preds_{target}': preds, f'model_pred_{target}': pred_idx ,f'correct_{target}': 1 if pred_idx == target else 0})

json.dump(results, open(filename, 'w'), indent=2)
