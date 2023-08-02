#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from model.model import GAT
from functions.load_data import load_data, accuracy
import torch.nn.functional as F

import csv
import yaml



#Read in Values from YAML File 
with open('./params/params.yaml') as params:
    param_dict = yaml.full_load(params)

#Store Values in Variables for use later
csv_file = param_dict["csv_file"]
disable_cuda = param_dict["disable_cuda"]
fastmode = param_dict["fastmode"]
random_seed = param_dict["random_seed"]
max_epochs = param_dict["max_epochs"]
learning_rate = param_dict["learning_rate"]
weight_decay = param_dict["weight_decay"]
hidden = param_dict["hidden"]
nb_heads = param_dict["nb_heads"]
dropout = param_dict["dropout"]
alpha = param_dict["alpha"]
patience = param_dict["patience"]
momentum = param_dict["momentum"]
optimizer_type = param_dict["optimizer_type"]

#Open CSV Reader
f = open(csv_file, 'w')
fieldnames = ['Epoch', 'loss_train', 'acc_val', 'loss_val', 'acc_val']
writer = csv.writer(f)
writer.writerow(fieldnames) 

#Setting up Cuda
cuda = not disable_cuda and torch.cuda.is_available()
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if cuda:
    torch.cuda.manual_seed(random_seed)

#Load Data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

#Create GAT model
model = GAT(nfeat=features.shape[1], nhid=hidden, nclass=int(labels.max()) + 1, dropout=dropout, nheads=nb_heads, alpha=alpha)

#Define Optimizer 
if optimizer_type == "ADAM":
    print("Using ADAM Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #lr = 0.05
elif optimizer_type == "Adadelta":
    print("Using Adadelta Optimizer")
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
elif optimizer_type == "SGD":
    print("Using SGD Optimizer")
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) 
else:
    print("Optimizer Type Not Recognized")


#Load Parameters onto CUDA 
if cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

#Define Loss Function Here
def compute_loss(output, labels):
    # original (negative-logliklihood loss):
    loss_value = F.nll_loss(output, labels) #original (negative-logliklihood loss)

    #CrossEntropyLoss
    # cross_entropy_loss = nn.CrossEntropyLoss()
    # loss_value = cross_entropy_loss(output, labels)
    return loss_value

#Define Training procedure
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = compute_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    loss_val = compute_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    row = [epoch+1,loss_train.data.item(), acc_train.data.item(), loss_val.data.item(), acc_val.data.item()]
    writer.writerow(row)

    return loss_val.data.item()

#Define Testing Procedure
def compute_test():
    model.eval()
    output = model(features, adj)
    #loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    loss_test = compute_loss(output[idx_train], labels[idx_train])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = max_epochs + 1
best_epoch = 0
for epoch in range(max_epochs):
    loss_values.append(train(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == patience:
        break

    
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
compute_test()

f.close()
