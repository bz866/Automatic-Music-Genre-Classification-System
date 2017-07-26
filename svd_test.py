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

num_tags = 50000

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

print('creating y matirx')

start =time.time()

# Convert indices to binary vectors of tags
y = np.zeros((len(indices), num_tags))
for i, tags in enumerate(indices):
    for tag in tags:
        y[i, tag] = 1


from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
print(time.time() - start)

print('converting to sparse')
y_sparse = csr_matrix(y)

print(time.time() - start)

print('starting svd')
svd = TruncatedSVD(n_components=100)
svd.fit(y_sparse)

print(svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum()) 
print(time.time() - start)


'''

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

#pickle.dump(model,open('nb.p','wb'))
'''
'''

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
