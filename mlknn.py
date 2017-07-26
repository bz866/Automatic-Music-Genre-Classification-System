from sklearn.neighbors import NearestNeighbors
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
from sklearn.metrics import f1_score
import time

from sklearn.metrics import f1_score


class ML_KNN():

    def __init__(self, k, num_labels):

        self.nn = NearestNeighbors(n_neighbors=k, radius=1.0)
        self.k = k
        self.num_labels = num_labels

    def fit(self, X, y):

        # identify prior probabilities
        counts = y_train.sum(axis=0)
        self.H = (counts + 1) / (counts + 1).sum(axis=0).sum()

        # Fit posterior probabilities
        self.nn.fit(X)
        N = self.nn.kneighbors(X)

        c1 = np.zeros((self.k + 1, self.num_labels))
        c0 = np.zeros((self.k + 1, self.num_labels))

        for i in range(500):
            delta = y_train[N[1][i]].sum(axis=0).astype(int)

            mask = np.nonzero(y[i])
            mask_neg = np.nonzero(1 - y[i])

            c1[tuple(delta[mask]), tuple(mask[0])] += 1
            c0[tuple(delta[mask_neg]), tuple(mask_neg[0])] += 1

        self.E_H1 = (c1 + 1) / (c1 + 1).sum(axis=0)
        self.E_H0 = (c0 + 1) / (c0 + 1).sum(axis=0)

    def predict(self, X):

        N_test = self.nn.kneighbors(X)

        y_hat = np.zeros((X.shape[0], self.num_labels))

        for i in range(X.shape[0]):
            count = y_train[N_test[1][i]].sum(axis=0).astype(int)

            prob_hasLabel = self.E_H1[tuple(count), tuple(range(100))] * self.H
            prob_noLabel = self.E_H0[tuple(count), tuple(range(100))] * (1 - self.H)

            y_hat[i] = 1 * (prob_hasLabel > prob_noLabel)

        return y_hat

    def score(self, X, y):

        y_hat = self.predict(X)

        hamming = hamming_loss(y, y_hat)

        f = f1_score(y, y_hat, average='micro')

        return (hamming, f)

start =time.time()

n = 20000

num_tags = 100

#load all lyric data into pandas dataframe
df = pd.read_csv('lyric_data.csv', index_col=0)

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
X = vect.fit_transform(df['lyrics'])
vocab = vect.vocabulary_
tok = vect.build_analyzer()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model =ML_KNN(5,100)
model.fit(X_train[:50000],y_train[:50000])

print(model.score(X_train[:50000],y_train[:50000]))
print(model.score(X_test,y_test))

