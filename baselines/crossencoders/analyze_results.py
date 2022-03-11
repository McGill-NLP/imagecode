from torch.nn.functional import softmax
import torch
import json
import sys

from collections import defaultdict
counter2key = json.load(open('counter2key_test.json', 'r'))
results = json.load(open(sys.argv[1], 'r'))
groups = defaultdict(list)

final = dict()

for pred in results:
    q_id = str(pred['question_id'])
    alignment = pred['prediction_aligned']
    not_alignment = pred['prediction_notaligned']
    p1, p2 = softmax(torch.Tensor([alignment, not_alignment]))
    p1 = p1.item()
    folder = q_id[:-2]
    groups[folder].append([q_id, p1])

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

print(total)
print(total_corr)
print(total_corr / total)
