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

start = time.time()

n = 20000

num_tags = 1000

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
X = vect.fit_transform(df['lyrics'])
vocab = vect.vocabulary_
tok = vect.build_analyzer()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


for alpha in range(1,10):

    model = OneVsRestClassifier(MultinomialNB(alpha=alpha)).fit(X_train, y_train)

    y_hat = model.predict_proba(X_test)

    def evaluate(y, y_hat):
        correct = y.flatten()
        predicted = y_hat.flatten()

        precision, recall, thresholds = precision_recall_curve(correct, predicted)
        f_score = 2 * precision * recall / (precision + recall)

        i_max = f_score.argmax()
        f_max = f_score[i_max]
        max_thresh = thresholds[i_max]

        hamming = hamming_loss(y, y_hat > max_thresh)

        return  f_max, hamming


    f_max_train, hamming_train= evaluate(y_train, model.predict_proba(X_train))

    f_max_test, hamming_test = evaluate(y_test, model.predict_proba(X_test))

    print('\nSmoothing Param ' , alpha )

    print('\nTRAINING ACCURACY')
    print("F Score: ", f_max_train)
    print("Hamming Loss: ", hamming_train)

    print("\nTESTING ACCURACY")
    print("F Score: ", f_max_test)
    print("Hamming Loss: ", hamming_test)

