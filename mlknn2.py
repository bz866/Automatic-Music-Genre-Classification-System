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
from scipy.spatial.distance import pdist
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import pairwise_distances

class ML_KNN():

    def __init__(self, k, num_labels):

        self.nn = NearestNeighbors(n_neighbors=k, radius=1.0)
        self.k = k
        self.num_labels = num_labels

    def nearest_neighbors(self,X,X_test=None):

        if X_test is not None:
            D = pairwise_distances(X_test,X)
        else:
            D = pairwise_distances(X)

        neighbors = D.argpartition( self.k , axis=1)[:,:self.k]
        return(neighbors)

    def fit(self, X, y):

        # identify prior probabilities
        counts = y.sum(axis=0)
        self.H = (counts + 1) / (counts + 1).sum(axis=0).sum()

        # Fit posterior probabilities
        #self.nn.fit(X)
        #N = self.nn.kneighbors(X)
        N = self.nearest_neighbors(X)

        c1 = np.zeros((self.k + 1, self.num_labels))
        c0 = np.zeros((self.k + 1, self.num_labels))

        for i in range(X.shape[0]):
            delta = y[N[i]].sum(axis=0).astype(int)

            mask = np.nonzero(y[i])
            mask_neg = np.nonzero(1 - y[i])

            c1[tuple(delta[mask]), tuple(mask[0])] += 1
            c0[tuple(delta[mask_neg]), tuple(mask_neg[0])] += 1

        self.E_H1 = (c1 + 1) / (c1 + 1).sum(axis=0)
        self.E_H0 = (c0 + 1) / (c0 + 1).sum(axis=0)
        self.X_train = X
        self.y_train = y

    def predict(self, X):

        #N_test = self.nn.kneighbors(X)
        N = self.nearest_neighbors(self.X_train,X)

        y_hat = np.zeros((X.shape[0], self.num_labels))

        for i in range(X.shape[0]):

            count = self.y_train[N[i]].sum(axis=0).astype(int)

            prob_hasLabel = self.E_H1[tuple(count), tuple(range(self.num_labels))] * self.H
            prob_noLabel = self.E_H0[tuple(count), tuple(range(self.num_labels))] * (1 - self.H)

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

np.save('X.npy' ,X)
np.save('y.npy',y)

p=.15

idx= np.random.permutation(X.shape[0])
idx = idx[int(np.floor(X.shape[0]*p )) : ]

X =X[idx]
y =y[idx]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model =ML_KNN(5,100)
model.fit(X_train,y_train)

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

