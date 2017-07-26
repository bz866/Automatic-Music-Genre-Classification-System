import pandas as pd
import numpy as np
from utils import *
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import hamming_loss
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from scipy.sparse import csr_matrix
from scipy import sparse

X_song = sparse.load_npz('X_song.npy')
X_tag = np.load('X_label.npy')
y = np.load('y.npy')

W = nn.Linear(1, 1, bias=True)

opt = optim.SGD(W.parameters(), lr=0.001)

bce = torch.nn.BCELoss()


def train(X_song, X_tag, y):
    n_batches = 20
    epochs = 10
    avg_loss = 0

    for e in range(epochs):

        for i in range(n_batches):
            song_idx = np.random.permutation(X_song.shape[0])[0:64]
            tag_idx = np.random.permutation(X_tag.shape[0])[0:64]

            song_batch = Variable(torch.from_numpy(X_song[song_idx].astype(np.float32)))
            tag_batch = Variable(torch.from_numpy(X_tag[tag_idx].astype(np.float32)))
            target = Variable(torch.from_numpy(y[song_idx, tag_idx]).float())

            delta = (song_batch - tag_batch).abs().mean()

            y_hat = F.sigmoid(W(delta))

            opt.zero_grad()

            loss = bce(y_hat, target)
            loss.backward()
            opt.step()

            avg_loss += loss.data[0]

    print(avg_loss / n_batches / epochs)


def test(X_song, X_tag, y):
    n_batches = 20
    avg_loss = 0
    y_hat_all = np.zeros(n_batches * 64)
    y_all = np.zeros(n_batches * 64)

    for i in range(n_batches):
        song_idx = np.random.permutation(X_song.shape[0])[0:64]
        tag_idx = np.random.permutation(X_tag.shape[0])[0:64]

        song_batch = Variable(torch.from_numpy(X_song[song_idx].astype(np.float32)))
        tag_batch = Variable(torch.from_numpy(X_tag[tag_idx].astype(np.float32)))
        target = Variable(torch.from_numpy(y[song_idx, tag_idx]).float())

        delta = (song_batch - tag_batch).abs().mean()

        y_hat = F.sigmoid(W(delta))

        y_hat_all[i * 64:(i + 1) * 64] = y_hat.data.numpy().flatten()
        y_all[i * 64:(i + 1) * 64] = y[song_idx, tag_idx]

        loss = bce(y_hat, target)

        avg_loss += loss.data[0]

    precision, recall, thresholds = precision_recall_curve(y_all.flatten(), y_hat_all.flatten())

    f_score = 2 * precision * recall / (precision + recall)
    i_max = np.nanargmax(f_score)
    f_max = f_score[i_max]
    max_thresh = thresholds[i_max]

    print(f_max, avg_loss / n_batches)



def test2(X_song, X_tag, y):

    n_batches = 20
    avg_loss = 0
    y_hat_all = np.zeros(n_batches * 64)
    y_all = np.zeros(n_batches * 64)

    for i in range(n_batches):
        song_idx = np.random.permutation(X_song.shape[0])[0:64]
        tag_idx = np.random.permutation(X_tag.shape[0])[0:64]

        song_batch = X_song[song_idx]
        tag_batch = X_tag[tag_idx]
        target = y[song_idx, tag_idx]

        delta = (song_batch - tag_batch).abs().mean()
        y_hat = delta

        y_hat_all[i * 64:(i + 1) * 64] = y_hat.data.numpy().flatten()
        y_all[i * 64:(i + 1) * 64] = y[song_idx, tag_idx]


    print(np.corrcoef(y_all,y_hat_all)[0,1])

    np.save('predicted.npy',y_hat_all)
    np.save('actual.npy', y_all)

print(X_song)
print(X_tag.shape,y.shape)
test(X_song,X_tag,y)
#for i in range(5):
#    train(X, tag_features, y)
#    test(X, tag_features, y)
