import json
from collections import defaultdict
import sys
import numpy as np
import yaml

best_fine = json.load(open(sys.argv[1], 'r'))

img_acc = 0
img_total = 0
vid_acc = 0
vid_total = 0

for img_set, preds in best_fine.items():
    for k, v in preds.items():
        if 'correct_' in k:
            acc = v
            idx = k[-1]
            if 'open-images' in img_set:
                img_total += 1
                img_acc += acc
            else:
                vid_total += 1
                vid_acc += acc

print(f'Accuracy of CLIP on videos: {round(vid_acc/vid_total, 3)}')
print(f'Accuracy of CLIP on images: {round(img_acc/img_total, 3)}')
print(f'Accuracy: {round((img_acc+vid_acc)/(img_total+vid_total), 4)}')
