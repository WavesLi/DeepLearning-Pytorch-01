# -*- coding: utf-8 -*-

# @File    : tools.py
# @Date    : 2019-09-05
# @Author  : litao
# @Describe:
import os
from pathlib import Path
from Config.config import args
import numpy as np
import torch

def get_root_path():
    this_path = Path.cwd()
    root_path = this_path.parent
    return root_path

def path_merge(child_path):
    return Path(get_root_path(),child_path)


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
if __name__ == '__main__':
    print(Path(get_root_path(),arg.surname_csv))