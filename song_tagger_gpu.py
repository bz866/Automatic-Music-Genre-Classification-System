import pandas as pd
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
from networks import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import hamming_loss

np.set_printoptions(precision=2)
#import theano.sandbox.cuda.basic_ops as sbcuda

n = 20000
seq_len = 80
h = 64
num_tags = 100
batch_size = 64

print(h)
gpu = False

print("loading data")
start = time.time()
glove = np.load('glove.npy')

features = np.load('features.npy')
y = np.load('y.npy')

features = torch.from_numpy(features)
y = torch.from_numpy(y).float()
glove = torch.from_numpy(glove)


train_idx = int(np.floor(features.size()[0] * 8 / 10))

train_loader = torch.utils.data.TensorDataset(features[:train_idx ], y[:train_idx ])
test_loader = torch.utils.data.TensorDataset(features[train_idx :], y[train_idx :])


if not gpu:

    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=True)
else:

    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=True,pin_memory=True)


print(time.time() - start)
print("creating model")

load = False



if load:
    model = torch.load('model.p')
else:
    model = LSTM_Model(h,glove,num_tags)
    
    #model = CNN(glove ,y.size()[1],features.size()[1])

    #bidir = False
    #pool = False

    #print(bidir,pool)
    #model = BiConvGRU( h = 256,conv_feat=200, glove = glove, num_out = num_tags,bidirectional = bidir , pooling = pool)

params = model.params


opt = optim.Adam(params, lr=0.001)

bce = torch.nn.BCELoss()

if gpu:
    model.cuda()
    bce.cuda()


print(model)

def train(train_loader):

    model.train()

    start = time.time()
    avg_loss = 0
    i = 1
    for data, target in train_loader:

        data, target = Variable(data), Variable(target)

        if gpu:

            data = data.cuda()                               
            target = target.cuda()


        opt.zero_grad()

        y_hat = model.forward(data)

        loss = bce(y_hat, target)

        loss.backward()


        opt.step()

        avg_loss += loss
        i += 1

        if i % 700 == 0:
            print("averge loss: ", (avg_loss / i).data[0], " time elapsed:", time.time() - start)
            #print( "CUDA memory: "  ,sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024/1024 )

            #for p in params:
            #    print(list(p.size()), (p.data **2 ).mean())# p.data.norm())





def test(test_loader):

    model.eval()

    avg_loss = 0
    
    y_hat_all = np.zeros(y[train_idx:].numpy().shape)
    y_target = np.zeros(y[train_idx:].numpy().shape)

    i = 0

    for data, target in test_loader:

        data, target = Variable(data), Variable(target)

        if gpu:
            target = target.cuda() 
            data = data.cuda()

        y_hat = model.forward(data)

        loss = bce(y_hat, target)

        y_hat = y_hat.cpu().data.numpy()

        y_hat_all[i*data.size()[0]:(i+1)*data.size()[0]] = y_hat
        y_target[i*data.size()[0]:(i+1)*data.size()[0]] = target.data.numpy()

        i+=1

        avg_loss += loss


    #precision, recall, thresholds = precision_recall_curve(y[train_idx :].numpy().flatten(), y_hat_all.flatten())
    precision, recall, thresholds = precision_recall_curve(y_target.flatten(), y_hat_all.flatten())

    
    f_score = 2* precision * recall / (precision + recall)
    i_max = np.nanargmax(f_score)
    f_max = f_score[i_max]
    max_thresh = thresholds[i_max]

    #hamming = hamming_loss(y[train_idx :].numpy(),y_hat_all > max_thresh)
    hamming = hamming_loss(y_target,y_hat_all > max_thresh)

    print("averge loss: ", (avg_loss / len(test_loader)).data[0], " average f score: ", f_max, " average hamming loss: ", hamming)
    print('precision: '  , precision[i_max], 'recall: ' , recall[i_max])



for i in range(20):
    
    train(train_loader)
    test(test_loader)

    torch.save(model,open('model.p','wb'))
