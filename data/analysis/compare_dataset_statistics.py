import matplotlib.pyplot as plt
import matplotlib
import json
from collections import defaultdict
import spacy
from collections import Counter

import yaml
import seaborn
seaborn.set()
seaborn.set_style('ticks')
# csfont = {'fontname':'Times New Roman'}
matplotlib.rc('font',family='Times New Roman')

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")

raw_dataset = json.load(open('dataset.json', 'r'))
train = json.load(open('train_data.json', 'r'))
valid = json.load(open('valid_data.json', 'r'))
test = json.load(open('test_data.json', 'r'))

nlvr2 = open('other_datasets_rawtext/all_data_NLVR2.txt', 'r').readlines()
spot = open('other_datasets_rawtext/all_data_spotdiff.txt', 'r').readlines()
cid = open('all_data.txt', 'r').readlines()
dataset = train | valid | test
# with open("all_data.txt", 'w') as f:
#     for k, v in dataset.items():
#         for idx, text in v.items():
#             f.write(text + '\n')

plt.xlabel('number of tokens')
plt.ylabel('% of descriptions')

for name, data in {'CID': cid, 'NLVR2': nlvr2, 'spot-the-diff': spot}.items():
    print("----------------------\n\n")
    print(name)
    sent_lengths = defaultdict(int)
    numb_toks = defaultdict(int)
    num_descr = 0
    count_tokens = 0
    types = set()
    dep_tree_depth = 0
    all_depths = []
    for text in data:
        doc = nlp(text.strip())
        # spacy.displacy.serve(doc, style="dep")
        depths = {}

        def walk_tree(node, depth):
            depths[node.orth_] = depth
            if node.n_lefts + node.n_rights > 0:
                return [walk_tree(child, depth + 1) for child in node.children]

        [walk_tree(sent.root, 0) for sent in doc.sents]
        # print(depths)
        dep_tree_depth += max(depths.values())
        all_depths.append((max(depths.values()), text.strip()))
        sent_lengths[len(list(doc.sents))] += 1
        count = 0
        for token in doc:
            if token.pos_ not in ['SPACE']:
                count += 1
                types.add((token.text.lower()))
        numb_toks[count] += 1
        # ann_valid[img_set][img_id]['number_tokens'] = str(count)
        # ann_valid[img_set][img_id]['number_sentences'] = str(len(list(doc.sents)))
        # ann_valid[img_set][img_id]['max_dependency_depth'] = str(max(depths.values()))
        num_descr += 1
        count_tokens += count

    # yaml.dump(ann_valid, open('ann_valid_data_rich.yaml', 'w'), default_style='"', sort_keys=False)

    print(f'Distrubtion of number of sentences per description: {sent_lengths}')
    descrs = 0
    sents = 0
    for k,v in sent_lengths.items():
        descrs += v
        sents += k * v
    print(f'Avg sentences per descr: {sents/descrs}')
    print(f'Average tokens per description: {count_tokens / num_descr}')
    print(f'Distrubtion of number of words per description: {numb_toks}')
    print(f'Number of types in dataset {len(types)}')
    print(f'Average dependency tree depth: {dep_tree_depth / num_descr}')
    all_depths = sorted(all_depths, key= lambda x: x[0], reverse=True)
    print(f'Top 20 depths {all_depths[:20]}')
    numb_toks = {x[0]: x[1] / num_descr for x in numb_toks.items()}
    numb_toks = sorted(numb_toks.items(), key=lambda x: x[0])
    x = [x[0] for x in numb_toks]
    y = [x[1] for x in numb_toks]

    plt.plot(x, y, label=name)
    plt.legend(loc='upper right', shadow=True, fontsize='medium')
    plt.grid()

    plt.savefig("numb_tokens.png")
