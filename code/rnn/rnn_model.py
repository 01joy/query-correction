#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:47:50 2019

@author: bytedance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np


torch.manual_seed(1)


txt=r'/Users/bytedance/Documents/datasets/search_word.txt'

xs=[]
ys=[]

fin=open(txt)
for line in fin:
    y,x = line.strip().split('\t') # 第一列是对的，第二列是错的
    xs.append(x)
    ys.append(y)

fin.close()

xs=xs[:100]
ys=ys[:100]

X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, random_state=42)

word_to_ix = {'SOS':0, 'EOS':1, 'UNK':2}		# 'SOS': start of sentencex
ix_to_word = {0:'SOS', 1:'EOS', 2:'UNK'}		# 'EOS': end of sentence

for sent, tags in zip(X_train,y_train):
    for w1,w2 in zip(sent,tags):
        if w1 not in word_to_ix:
            cid = len(word_to_ix)
            word_to_ix[w1] = cid
            ix_to_word[cid] = w1
        if w2 not in word_to_ix:
            cid = len(word_to_ix)
            word_to_ix[w2] = cid
            ix_to_word[cid] = w2
#print(word_to_ix)
#print(ix_to_word)

def prepare_sequence(seq, to_ix):
    idxs = [0]
    for w in seq:
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(2) # unknown
    idxs.append(1)
    return torch.tensor(idxs, dtype=torch.long)


def get_sequence(scores, to_word):
    idxs = scores.argmax(dim=1, keepdim=True)
    words=[]
    for i in idxs:
        words.append(to_word[i.item()])
    return words
    
    
# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 100
HIDDEN_DIM = 200


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    
    
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(word_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
print('query: %s\n'%X_train[0])
with torch.no_grad():
    inputs = prepare_sequence(X_train[0], word_to_ix)
    tag_scores = model(inputs)
    print('initial correction: %s\n'%get_sequence(tag_scores,ix_to_word))

nepoch=300
train_loss = []
test_losss = []
for epoch in range(nepoch):  # again, normally you would NOT do 300 epochs, it is toy data
    cur_train_loss = 0
    cur_test_loss = 0
    for sentence, tags in zip(X_train, y_train):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, word_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
#        print('id=%d\t1sz=%d\t2sz=%d'%(i,len(tag_scores),len(targets)))
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        cur_train_loss += loss.item()
    cur_train_loss /= len(X_train)


    with torch.no_grad():
        for sentence, tags in zip(X_test, y_test):
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, word_to_ix)
            tag_scores = model(sentence_in)
            cur_test_loss += loss_function(tag_scores, targets)
        cur_test_loss /= len(X_test)
    print('epoch=%d/%d,train_loss=%f,test_loss=%f'%(epoch,nepoch,cur_train_loss,cur_test_loss))

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(X_train[0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print('trained correction: %s\n'%get_sequence(tag_scores,ix_to_word))
