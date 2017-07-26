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
from sklearn.metrics import roc_auc_score
import time
import sys
from utils import *

n = 20000
seq_len = 80
h = 128
num_tags = 100
batch_size = 64


dtype = torch.cuda.FloatTensor

# load all lyric data into pandas dataframe
df = pd.read_csv('lyric_data.csv', index_col=0)#.iloc[0:200000]
#df = pd.read_csv('lyric_data_small.csv', index_col=0)


# Sometimes the API returns an error message rather than actual lyrics. This removes it
bad_song = df['lyrics'].value_counts().index[0]
df[df['lyrics'] == bad_song] = ''

# only take the ones that we have data for
df.fillna('', inplace=True)
df = df[df['lyrics'] != '']

# List of list of tags for each example
tags = [clean_tags(raw) for raw in list(df['tags'])]

# list of tuples of (tag, frequency) in desending order
tf = tag_freq(tags)

# Choose which tags to restrict model too
important_tags = [x[0] for x in tf[0:num_tags]]
important_tags = dict(zip(important_tags, range(len(important_tags))))

# maps each of the tags int 'tags' to an int index
indices = tag2index(tags, important_tags)

# Convert indices to binary vectors of tags
y = np.zeros((len(indices), num_tags))
for i, tags in enumerate(indices):
    for tag in tags:
        y[i, tag] = 1

# Build vocabulary and tokenizer
vect = CountVectorizer(max_features=n, stop_words='english')
vect.fit(df['lyrics'])
vocab = vect.vocabulary_
tok = vect.build_analyzer()

# Load glove vectors for word embedding
vocab, glove = load_glove(vocab)


# Convert text to sequence input
features = df['lyrics'].apply(lambda x: sent2seq(x, vocab, tok, seq_len))
features = np.array(list(features))

shuffle = np.random.permutation(features.shape[0])


np.save('features.npy',features[shuffle])
np.save('y.npy',y[shuffle])
np.save('glove.npy',glove)

