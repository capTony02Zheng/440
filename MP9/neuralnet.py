# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP9. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        ## code here
        self.net = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=out_size)
        )
        self.optimizer = optim.SGD(self.net.parameters(), lr=lrate)
        
        #
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        ## code here
        x = self.net(x)
        return x
        return torch.ones(x.shape[0], 1)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        
        ## code here
        

        outputs = self.forward(x)
        # print(outputs.size())
        # print(y.size())
        loss = self.loss_fn(outputs, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # loss = self.loss_fn(self.forward(x), y)
        # loss.backward()

        return loss.item()



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    # dataset = get_dataset_from_arrays()
    ## standarize TODO
    def standardize_data(train_set, dev_set):
        """
        Standardize the training and development data.

        @param train_set: an (N, in_size) Tensor
        @param dev_set: an (M, in_size) Tensor

        @return standardized_train_set: Standardized training data
        @return standardized_dev_set: Standardized development data
        """
        mean = train_set.mean(0)
        std = train_set.std(0)
        standardized_train_set = (train_set - mean) / std
        standardized_dev_set = (dev_set - mean) / std
        return standardized_train_set, standardized_dev_set
    train_set, dev_set = standardize_data(train_set, dev_set)

    trainloader = torch.utils.data.DataLoader(get_dataset_from_arrays(train_set,train_labels), batch_size=batch_size,shuffle=False, num_workers=2)


    # __init__(self, lrate, loss_fn, in_size, out_size)
    net = NeuralNet(0.01, torch.nn.CrossEntropyLoss() ,train_set.size(1),4)
    losses = []

    for i in range(epochs):
        total_loss = 0.0
        for j, data in enumerate(trainloader, 0):
            # print(data)
            # print(i)
            # print()
            x_batch = data["features"]
            y_batch = data["labels"]
            # break
            loss = net.step(x_batch, y_batch)
            total_loss += loss

            # forward + backward + optimize
            # outputs = net(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
        avg_loss = total_loss / len(train_set)
        losses.append(avg_loss)
        print("finished epoch ", i)
        break

    network = net(dev_set).detach().cpu().numpy()
    yhats = np.argmax(network,axis=1)
    # for i in range(len(yhats)):
        

    # print(losses)
    # print(yhats)
    return losses, yhats, net
    # return [],[],None
