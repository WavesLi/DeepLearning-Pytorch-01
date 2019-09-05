# -*- coding: utf-8 -*-

# @File    : config.py
# @Date    : 2019-09-05
# @Author  : litao
# @Describe:

from argparse import Namespace

args = Namespace(
    # Data and path information
    surname_csv="Corpus/surnames/surnames_with_splits.csv",
    vectorizer_file="Model/vectorizer.json",
    model_state_file="Model/surname_classification/model.pth",
    save_dir="Model/surname_classification/",
    # Model hyper parameter
    char_embedding_size=100,
    rnn_hidden_size=64,
    # Training hyper parameter
    num_epochs=100,
    learning_rate=1e-3,
    batch_size=64,
    seed=1337,
    early_stopping_criteria=5,
    # Runtime hyper parameter
    device="cpu",
    cuda=False,
    catch_keyboard_interrupt=True,
    reload_from_files=False,
    expand_filepaths_to_save_dir=True,
)