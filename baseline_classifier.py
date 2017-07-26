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
vect = CountVectorizer(max_features=n, stop_words='english' , ngram_range=(1, 2))
X = vect.fit_transform(df['lyrics'])
vocab = vect.vocabulary_
tok = vect.build_analyzer()

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model = OneVsRestClassifier(MultinomialNB()).fit(X_train,y_train)
model = OneVsRestClassifier(LogisticRegression(C=1)).fit(X_train,y_train)

y_hat = model.predict_proba(X_test)

def evaluate(y,y_hat):

    correct = y.flatten()
    predicted = y_hat.flatten()

    auc = roc_auc_score(correct, predicted)

    precision, recall, thresholds = precision_recall_curve(correct, predicted)
    f_score = 2* precision * recall / (precision + recall)

    i_max = f_score.argmax()
    f_max = f_score[i_max]
    max_thresh  =  thresholds[i_max]

    hamming = hamming_loss(y,y_hat > max_thresh)

    return auc,f_max,hamming,max_thresh

auc_train,f_max_train,hamming_train , max_thresh_train = evaluate(y_train, model.predict_proba(X_train))

auc_test,f_max_test,hamming_test , max_thresh_test = evaluate(y_test, model.predict_proba(X_test))

print('total time: ' , time.time() - start)

print('\nTRAINING ACCURACY')
print("AUC: " , auc_train)
print("F Score: " , f_max_train)
print("Hamming Loss: " , hamming_train)

print("\nTESTING ACCURACY")
print("AUC: " , auc_test)
print("F Score: " , f_max_test)
print("Hamming Loss: " , hamming_test)


#pickle.dump(model,open('nb.p','wb'))

'''
y_hat = model.predict_proba(X_test) > max_thresh_test

tags = [clean_tags(raw) for raw in list(df['tags'])]

index2tag = {v: k for k, v in important_tags.items()}

num_out = 0

#Print out examples of where we are way off for analysis
for i in range(X_test.shape[0]):

    if hamming_loss(y_test[i] , y_hat[i]) > hamming_test:
        
        print('SONG')
        print(df['artist'].iloc[i],df['song'].iloc[i])

        print('LYRICS')
        print(df['lyrics'].iloc[i])

        print('BAG OF WORDS')
        print(tok(df['lyrics'].iloc[i]))

        print('TAGS')
        print([ t for t in tags[i] if t in important_tags])

        print('PREDICTED TAGS')

        tag_idx = list(y_hat[i].nonzero()[0])
        pred_tags = [ index2tag[x] for x in tag_idx  ]

        print(pred_tags)
        print()
        print()

        num_out +=1 

        if num_out > 30:

            break

'''
