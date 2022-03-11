import json
import random

import numpy as np
import krippendorff

raw_dataset = json.load(open('dataset.json', 'r'))
train = json.load(open('train_data.json', 'r'))
valid = json.load(open('valid_data.json', 'r'))
test = json.load(open('test_data.json', 'r'))

matrix = []

# FOR TEST/VAL
for img_set, val in raw_dataset.items():
    if img_set not in test:
        continue
    for idx, info in val.items():
        rets = info['data']['retrieval']
        if len(rets) > 1:
            correct = 0
            for pred in rets:
                if pred == int(idx):
                    correct += 1
            if correct >= 2:
                if len(rets) == 2:
                    rets = np.array(rets + [np.nan])
                else:
                    rets = np.array(rets)
                matrix.append(rets)


matrix = np.array(matrix)
matrix = matrix.transpose([1, 0])
print(matrix.shape)
print(round(krippendorff.alpha(reliability_data=matrix, level_of_measurement='nominal'), 6))
print(round(krippendorff.alpha(reliability_data=matrix, level_of_measurement='interval'), 6))
