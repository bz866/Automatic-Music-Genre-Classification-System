import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle
import torchvision
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

def f_score(y, y_hat):

    correct = y.flatten()
    predicted = y_hat.flatten()

    precision, recall, thresholds = precision_recall_curve(correct, predicted)
    f_score = 2 * precision * recall / (precision + recall)

    i_max = np.nanargmax(f_score)

    f_max = f_score[i_max]

    return f_max


def tag_freq(tags):
    indiv_tf = {}

    for tag in tags:

        for t in tag:

            if t in indiv_tf:
                indiv_tf[t] += 1
            else:
                indiv_tf[t] = 1

    tf = sorted(list(indiv_tf.items()), key=lambda x: -x[1])

    return tf


def word_freq(lyrics, n):
    vect = CountVectorizer(max_features=n, stop_words='english')

    bag = vect.fit_transform(lyrics).toarray()

    freq = list(bag.sum(axis=0))

    word_freq = [(word, f) for word, f in zip(list(vect.vocabulary_.keys()), freq)]

    word_freq = sorted(word_freq, key=lambda x: -x[1])

    return word_freq


def clean_tags(raw):
    if raw == '':
        return []

    tags = raw[1:-2].split("]")
    tags = [tag.split("'")[1] for tag in tags]
    return tags


def tag2index(tags, tag_map):
    indices = []

    for tag in tags:

        x = []
        for e in tag:

            if e in tag_map:
                x.append(tag_map[e])

        indices.append(x)
    return indices


def sent2seq(text, key, tok, l):
    words = tok(text)

    unknown = len(key.keys()) + 1

    seq = []
    for word in words:
        if word in key:
            seq.append(key[word] + 1)
        else:
            seq.append(unknown)

    if len(seq) > l:
        return seq[:l]
    else:
        padding = [0 for i in range(l - len(seq))]

        return (padding + seq)

    return seq


def load_glove(vocab):
    embedding_mat = [np.zeros(50)]
    new_vocab = {}

    count = 0

    with open('glove.6B.50d.txt') as f:

        for i, line in enumerate(f):

            s = line.split()

            if s[0] in vocab:
                embedding_mat.append(np.asarray(s[1:]))
                new_vocab[s[0]] = count
                count += 1

                if len(list(new_vocab.keys())) == len(list(vocab.keys())):
                    return new_vocab, np.array(embedding_mat)

    embedding_mat.append(np.random.randn(50))

    return new_vocab, np.array(embedding_mat).astype(np.float32())
