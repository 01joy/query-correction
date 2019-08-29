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

import sys
sys.path.append("..")

import config
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import argparse
from sklearn.utils import shuffle
from torch.autograd import Variable


def prepare_sequence(seq, to_ix):
    idxs = [to_ix['SOS']]
    for w in seq:
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(to_ix['UNK']) # unknown
    idxs.append(to_ix['EOS'])
    return idxs
    return torch.tensor(idxs, dtype=torch.long)


def get_sequence(scores, to_word):
    idxs = scores.argmax(dim=1, keepdim=True)
    words=[]
    for i in idxs:
        words.append(to_word[i.item()])
    return words


class BiLSTMCorrecter(nn.Module):

    def __init__(self, nb_lstm_layers, embedding_dim, hidden_dim, vocab_size, target_size, padding_idx):
        super(BiLSTMCorrecter, self).__init__()
        self.nb_lstm_layers = nb_lstm_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.target_size = target_size
        #self.batch_size = batch_size
        self.padding_idx = padding_idx
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, nb_lstm_layers, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def init_hidden(self):
        # the weights are of the form (nb_lstm_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.hidden_dim)
        hidden_b = torch.randn(self.nb_lstm_layers, self.batch_size, self.hidden_dim)

#        if self.hparams.on_gpu:
#            hidden_a = hidden_a.cuda()
#            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a).double()
        hidden_b = Variable(hidden_b).double()

        return (hidden_a, hidden_b)

    def forward(self, X, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.batch_size, seq_len = X.size()
        self.hidden = self.init_hidden()
        
        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = self.word_embeddings(X)

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.hidden2tag(X)

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        X = X.view(self.batch_size, seq_len, self.target_size)

        Y_hat = X
        return Y_hat

    def loss(self, Y_hat, Y, X_lengths):
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all the labels
        Y = Y.view(-1)

        # flatten all predictions
        Y_hat = Y_hat.view(-1, self.target_size)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        #tag_pad_token = self.tags['<PAD>']
        mask = (Y > self.padding_idx).double()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).item())

        # pick the values for the label and zero out the rest with the mask
        Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(Y_hat) / nb_tokens

        return ce_loss


    # def forward(self, sentences, sentence_length):
    #     # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
    #     # a new batch as a continuation of a sequence
    #     self.hidden = self.init_hidden()

    #     batch_size, seq_len, _ = sentences.size()
        
    #     embeds = self.word_embedding(sentences)

    #     X = torch.nn.utils.rnn.pack_padded_sequence(embeds, sentence_length, batch_first=True)

    #     #embeds = self.word_embeddings(sentence)
    #     lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
    #     tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
    #     tag_scores = F.log_softmax(tag_space, dim=1)
    #     return tag_scores
    
    
#model = BiLSTMCorrecter(config.EMBEDDING_DIM, config.HIDDEN_DIM, len(word_to_ix), len(word_to_ix))
#loss_function = nn.NLLLoss()
#optimizer = optim.Adam(model.parameters(), lr=config.LR)


#for epoch in range(config.EPOCH_NUM):  # again, normally you would NOT do 300 epochs, it is toy data
#    for sentence, tags in zip(X_train, y_train):
#        # Step 1. Remember that Pytorch accumulates gradients.
#        # We need to clear them out before each instance
#        model.zero_grad()
#
#        # Step 2. Get our inputs ready for the network, that is, turn them into
#        # Tensors of word indices.
#        sentence_in = prepare_sequence(sentence, word_to_ix)
#        targets = prepare_sequence(tags, word_to_ix)
#
#        # Step 3. Run our forward pass.
#        tag_scores = model(sentence_in)
#
#        # Step 4. Compute the loss, gradients, and update the parameters by
#        #  calling optimizer.step()
##        print('id=%d\t1sz=%d\t2sz=%d'%(i,len(tag_scores),len(targets)))
#        loss = loss_function(tag_scores, targets)
#        loss.backward()
#        optimizer.step()
#    print('epoch=%d/%d'%(epoch,config.EPOCH_NUM))
#
#
#with torch.no_grad():
#    inputs = prepare_sequence(X_train[0], word_to_ix)
#    tag_scores = model(inputs)
#    print('trained correction: %s\n'%get_sequence(tag_scores,ix_to_word))
#    
#   

def preprocess_data(inputs, labels, padding_idx):
    inputs, labels = zip(*sorted(zip(inputs, labels),key=lambda x: len(x[0]), reverse=True))

    # get the length of each sentence
    X_lengths = [len(sentence) for sentence in inputs]
    # create an empty matrix with padding tokens
    #pad_token = vocab['<PAD>']
    longest_sent = max(X_lengths)
    cur_batch_size = len(inputs)
    padded_X = np.ones((cur_batch_size, longest_sent)) * padding_idx
    # copy over the actual sequences
    for i, x_len in enumerate(X_lengths):
        sequence = inputs[i]
        padded_X[i, 0:x_len] = sequence[:x_len]


    # get the length of each sentence
    Y_lengths = [len(sentence) for sentence in labels]
    # create an empty matrix with padding tokens
    longest_sent = max(Y_lengths)
    cur_batch_size = len(labels)
    padded_Y = np.ones((cur_batch_size, longest_sent)) * padding_idx
    # copy over the actual sequences
    for i, y_len in enumerate(Y_lengths):
        sequence = labels[i]
        padded_Y[i, 0:y_len] = sequence[:y_len]

    tensor_X_train = torch.stack([torch.tensor(i, dtype=torch.long) for i in padded_X]) # transform to torch tensors
    tensor_y_train = torch.stack([torch.tensor(i, dtype=torch.long) for i in padded_Y])

    return tensor_X_train, tensor_y_train, X_lengths


def train(args, model, device, X, y, optimizer, batch_size, epoch, target_size,padding_idx):
    X, y = shuffle(X, y)
    model.train()
    k = 0   
    while k < len(X):
        inputs = X[k:k+batch_size]
        labels = y[k:k+batch_size]

        # some computation
        k+= batch_size
        
        tensor_X_train, tensor_y_train, X_lengths = preprocess_data(inputs, labels, padding_idx)
        data, target = tensor_X_train.to(device), tensor_y_train.to(device)

        optimizer.zero_grad()
        output = model(data, X_lengths)
        #loss = F.nll_loss(output, target)
        loss = model.loss(output,target,X_lengths)

#        target = target.view(-1)
#        # flatten all predictions
#        output = output.view(-1, target_size)
#        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
#        if batch_idx % args.log_interval == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(data), len(train_loader.dataset),
#                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, X, y, batch_size, target_size, padding_idx, test_name):
    model.eval()
    test_loss = 0
    correct = 0
    y_true=[]
    y_pred=[]
    with torch.no_grad():
        X, y = shuffle(X, y)

        k = 0
        alllen = 0
        while k < len(X):
            inputs = X[k:k+batch_size]
            labels = y[k:k+batch_size]
            # some computation
            k+= batch_size

            tensor_X_train, tensor_y_train, X_lengths = preprocess_data(inputs, labels, padding_idx)
            data, target = tensor_X_train.to(device), tensor_y_train.to(device)

            output = model(data, X_lengths)
            #loss = F.nll_loss(output, target)
            test_loss += model.loss(output,target,X_lengths)*sum(X_lengths)

            alllen += sum(X_lengths)
#
#            target = target.view(-1)
#            # flatten all predictions
#            output = output.view(-1, target_size)
#            test_loss += F.nll_loss(output, target, reduction='sum').item()


        # for data, target in test_loader:
        #     data, target = data.to(device), target.to(device)
        #     output = model(data)
        #     test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        #     pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        #     correct += pred.eq(target.view_as(pred)).sum().item()
        #     y_true.append(target)
        #     y_pred.append(pred.view_as(target))

        test_loss /= alllen

    # print('\n{} set: Average loss: {:.4f}, Overall accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_name, test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    
    return test_loss

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    xs=[]
    ys=[]
    
    fin=open(config.big_generated_txt)
    for line in fin:
        y,x = line.strip().split('\t') # 第一列是对的，第二列是错的
        xs.append(x)
        ys.append(y)
    
    fin.close()
    
    xs=xs[:100]
    ys=ys[:100]
    
    xtrains, xtests, ytrains, ytests = train_test_split(xs, ys, test_size=0.2, random_state=42)
    
    word_to_ix = {'<PAD>':0, 'SOS':1, 'EOS':2, 'UNK':3 }		# 'SOS': start of sentencex
    ix_to_word = {0:'<PAD>', 1:'SOS', 2:'EOS', 3:'UNK' }		# 'EOS': end of sentence; '<PAD>': for batch padding
    
    for sent, tags in zip(xtrains,ytrains):
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
    
    X_train, X_test, y_train, y_test = [], [], [], []
    for sentence, tags in zip(xtrains, ytrains):
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, word_to_ix)
        X_train.append(sentence_in)
        y_train.append(targets)

    for sentence, tags in zip(xtests, ytests):
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, word_to_ix)
        X_test.append(sentence_in)
        y_test.append(targets)
#
#    tensor_X_train = torch.stack([torch.tensor(i) for i in X_train]) # transform to torch tensors
#    tensor_y_train = torch.stack([torch.tensor(i, dtype=torch.long) for i in y_train])
#    tensor_X_test = torch.stack([torch.tensor(i) for i in X_test]) # transform to torch tensors
#    tensor_y_test = torch.stack([torch.tensor(i, dtype=torch.long) for i in y_test])

#    tensor_X_train=tensor_X_train.unsqueeze(1)
#    tensor_X_test=tensor_X_test.unsqueeze(1)
#
#    train_dataset = data_utils.TensorDataset(tensor_X_train, tensor_y_train)
#    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
#
#    test_dataset = data_utils.TensorDataset(tensor_X_test, tensor_y_test)
#    test_loader = data_utils.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    model = BiLSTMCorrecter(1, config.EMBEDDING_DIM, config.HIDDEN_DIM, len(word_to_ix), len(word_to_ix), word_to_ix['<PAD>']).to(device).double()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    train_loss=[]
    test_loss=[]
    for epoch in range(1, args.epochs + 1):
        print('epoch=%d'%epoch)
        train(args, model, device, X_train, y_train, optimizer, args.batch_size, epoch, len(word_to_ix), word_to_ix['<PAD>'])
        train_loss.append(test(args, model, device, X_train, y_train, args.batch_size, len(word_to_ix), word_to_ix['<PAD>'], 'Train'))
        test_loss.append(test(args, model, device, X_test, y_test, args.batch_size, len(word_to_ix), word_to_ix['<PAD>'], 'Test'))

    # train_loss_final, y_train_true, y_train_pred = test(args, model, device, train_loader, 'Train')
    # test_loss_final, y_test_true, y_test_pred = test(args, model, device, test_loader, 'Test')

    model_name = 'BiLSTMCorrecter'

    fig = plt.figure()
    plt.plot(train_loss,label='train_loss')
    plt.plot(test_loss,label='test_loss')
    plt.legend()
    plt.show()
    fig.savefig(r'%s.png'%model_name)

    if (args.save_model):
        torch.save(model.state_dict(),"%s.pt"%model_name)

if __name__ == '__main__':
    main()