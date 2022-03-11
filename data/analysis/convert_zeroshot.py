import json
from collections import defaultdict
import torch
from torch.nn.functional import softmax
import sys

data = list(json.load(open(sys.argv[1], 'r')))
mapping = json.load(open('counter2key_val.json', 'r'))
ids = json.load(open('shortid2id.json', 'r'))

results = defaultdict(dict)
for i in range(0,len(data)-10, 10):
    real_id = ids[mapping[str(data[i]['question_id'])]]
    real_id = real_id.split('___')[0]
    target = int(str(data[i]['question_id'])[-2])

    preds = []
    pred_idx = 0
    best_pred = 0
    for j in range(10):
        pred1 = data[i+j]['prediction_aligned']
        pred2 = data[i+j]['prediction_notaligned']
        p1, p2 = softmax(torch.Tensor([pred1, pred2]))
        preds.append(p1.item())
        if p1 > best_pred:
            best_pred = p1
            pred_idx = j

    results[real_id].update({f'raw_preds_{target}': preds, f'model_pred_{target}': pred_idx ,f'correct_{target}': 1 if pred_idx == target else 0})

json.dump(results, open(sys.argv[1], 'w'), indent=2)
