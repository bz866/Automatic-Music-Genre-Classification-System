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
import scipy
from scipy import sparse

n = 20000

num_tags = 1000

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
#X = X.toarray()
vocab = vect.vocabulary_
tok = vect.build_analyzer()

tag_features = np.zeros((num_tags, n))

for i in range(num_tags):

    mask = y[:, i] == 1

    tag_features[i] = X[mask].mean(axis=0)

    if i % 100 == 0:
        print(i)

#sparse.save_npz('X_song.npy',X)

np.save('X_song.npy',X.toarray())
np.save('X_label.npy' , tag_features)
np.save('y.npy',y)
