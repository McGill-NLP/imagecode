import json

best_fine = json.load(open('results/clip/NOCONTRA_clip_valid_set.json', 'r'))
ann_split = json.load(open('annotator_split_valid.json', 'r'))

acc_unseen = 0
total_unseen = 0
acc_seen = 0
total_seen = 0

for img_set, preds in best_fine.items():
    for k, v in preds.items():
        if 'correct_' in k:
            acc = v
            idx = k[-1]
            worker_type = ann_split[img_set][idx]
            if worker_type == 'unseen_worker':
                acc_unseen += acc
                total_unseen += 1
            elif worker_type == 'train_worker':
                acc_seen += acc
                total_seen += 1
            else:
                print('wtf')

print(f'Performance on unseen workers: {round(acc_unseen/total_unseen,3)}')
print(f'Performance on seen workers: {round(acc_seen/total_seen,3)}')