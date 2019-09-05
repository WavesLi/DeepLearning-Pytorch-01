# -*- coding: utf-8 -*-

# @File    : predict_process.py
# @Date    : 2019-09-05
# @Author  : litao
# @Describe:
import torch
from Utils.dataset import SurnameDataset
from Config.config import args
from NeuralNetwork.Net import SurnameClassifier
from Utils.tools import path_merge


dataset = SurnameDataset.load_dataset_and_load_vectorizer(args.surname_csv,args.vectorizer_file)
vectorizer = dataset.get_vectorizer()
classifier = SurnameClassifier(embedding_size=args.char_embedding_size,
                               num_embeddings=len(vectorizer.char_vocab),
                               num_classes=len(vectorizer.nationality_vocab),
                               rnn_hidden_size=args.rnn_hidden_size,
                               padding_idx=vectorizer.char_vocab.mask_index)
classifier.load_state_dict(torch.load(path_merge(args.model_state_file)))


def predict_nationality(surname, classifier, vectorizer):
    vectorized_surname, vec_length = vectorizer.vectorize(surname)
    vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(dim=0)
    vec_length = torch.tensor([vec_length], dtype=torch.int64)

    result = classifier(vectorized_surname, vec_length, apply_softmax=True)
    probability_values, indices = result.max(dim=1)

    index = indices.item()
    prob_value = probability_values.item()

    predicted_nationality = vectorizer.nationality_vocab.lookup_index(index)

    return {'nationality': predicted_nationality, 'probability': prob_value, 'surname': surname}

for surname in ['McMahan', 'Nakamoto', 'Wan', 'Cho',"LiTao","TaoLi"]:
    print(predict_nationality(surname, classifier, vectorizer))