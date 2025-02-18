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
This is the main entry point for MP10. You should only modify code within this file.
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

        # self.cnn = nn.Sequential(nn.Conv2d(3,9,kernel_size=4),nn.ReLU())
        # self.cnnT =nn.Sequential(nn.Conv2d(9,3,kernel_size=4),nn.ReLU())

        self.net = nn.Sequential(
            nn.Conv2d(3,16,8,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,8,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.2),
            nn.Flatten(), 
            nn.Linear(2048, 64),
            nn.LeakyReLU(),
            nn.Linear(64, out_size)
        )
        self.optimizer = optim.SGD(self.net.parameters(), lr=lrate, momentum=0.09)

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # 2883000 = 31 * 31 * 3000
        # 3000 images
        # print(x.size())
        x = x.view(-1,3,31,31)
        return self.net(x)
        return torch.ones(x.shape[0], 1)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        regularization_strength = 0.03 # can be modified value
        l2_reg =0.0
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)

        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        loss += 0.5 * regularization_strength * l2_reg
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        return 0.0



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
    net = NeuralNet(0.025, torch.nn.CrossEntropyLoss() ,train_set.size(1),4)
    losses = []

    for i in range(epochs):
        total_loss = 0.0
        for j, data in enumerate(trainloader, 0):
            x_batch = data["features"]
            y_batch = data["labels"]
            loss = net.step(x_batch, y_batch)
            total_loss += loss

        avg_loss = total_loss / len(train_set)
        losses.append(avg_loss)
        print("finished epoch ", i, "with loss ", avg_loss)
        # break

    network = net(dev_set).detach().cpu().numpy()
    yhats = np.argmax(network,axis=1)
        

    # print(losses)
    # print(yhats)
    return losses, yhats, net
    return [],[],None
