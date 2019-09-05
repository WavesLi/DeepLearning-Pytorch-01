# -*- coding: utf-8 -*-

# @File    : predeal.py
# @Date    : 2019-09-05
# @Author  : litao
# @Describe:


import pandas as pd
import numpy as np
from argparse import Namespace
import collections

args =  Namespace(
    raw_dataset_csv="surnames/surnames.csv",
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="surnames/surnames_with_splits.csv",
    seed=1337
)

surnames_pd = pd.read_csv(args.raw_dataset_csv,header=0)

nationality_map = {item:index for index,item in enumerate(list(set(surnames_pd.nationality.to_list())))}
# Splitting train by nationality
# Create dict
#对数据按国籍进行分组
by_nationality = collections.defaultdict(list)
for _, row in surnames_pd.iterrows():
    by_nationality[row.nationality].append(row.to_dict())


#按照数据样本的分割比例对数据进行分割
# Create split data
final_list = []
np.random.seed(args.seed)
for _, item_list in sorted(by_nationality.items()):
    np.random.shuffle(item_list)
    n = len(item_list)
    n_train = int(args.train_proportion * n)
    n_val = int(args.val_proportion * n)
    n_test = int(args.test_proportion * n)

    # Give data point a split attribute
    for item in item_list[:n_train]:
        item['split'] = 'train'
        item["nationality_index"] = nationality_map[item["nationality"]]
    for item in item_list[n_train:n_train + n_val]:
        item['split'] = 'val'
        item["nationality_index"] = nationality_map[item["nationality"]]

    for item in item_list[n_train + n_val:]:
        item['split'] = 'test'
        item["nationality_index"] = nationality_map[item["nationality"]]

        # Add to final list
    final_list.extend(item_list)

final_surnames = pd.DataFrame(final_list)
final_surnames.to_csv(args.output_munged_csv, index=False)