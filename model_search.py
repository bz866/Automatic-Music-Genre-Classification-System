import pandas as pd
import numpy as np
from utils import *
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import hamming_loss
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

start = time.time()

n = 20000

num_tags = 100

# load all lyric data into pandas dataframe
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
tfidf = TfidfVectorizer(max_features=n, stop_words='english')

X = tfidf.fit_transform(df['lyrics'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

# NAIVE BAYES
model = OneVsRestClassifier(MultinomialNB())
model.fit(X_train, y_train)

results.append(('Naive Bayes', model.score(X_test, y_test), f_score(y_test, model.predict_proba(X_test))) )

# LINEAR SVM

C = [.01, .1, 1, 10]
for c in C:
    start = time.time()
    model = OneVsRestClassifier(LinearSVC(max_iter=4, C=c))
    model.fit(X_train, y_train)
    results.append(('SVM ' + str(c), model.score(X_test, y_test), f1_score(y_test.flatten(), model.predict(X_test).flatten())))

C = [.01, .1, 1, 10]
# LOGISTIC REGRESSION
for c in C:
    start = time.time()
    model = OneVsRestClassifier(LogisticRegression(max_iter=4, C=c))
    model.fit(X_train, y_train)
    results.append(('Logistic Regresion ' + str(c) , model.score(X_test, y_test), f_score(y_test, model.predict_proba(X_test))))

#pd.DataFrame(results).to_csv('results_bow.csv')
pd.DataFrame(results).to_csv('results_tfidf.csv')


