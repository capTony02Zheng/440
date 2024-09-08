# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
import numpy as np
import operator
from collections import defaultdict
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""

"""

"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=0.01, pos_prior=0.95, silently=False):
    print_values(laplace,pos_prior)
    # training set is a list of lists of words (each list of words contains all the words in one review)
    # labels (positive or negative review) for the training set
    # list of 0's (Negative) and 1's (Positive)
    # ##
    index = 0;
    pos = []
    neg = []
    # c_pos = 0
    # c_all = 0
    for doc in train_set:
        if (train_labels[index] == 0):
            neg += doc
        else:
            pos += doc
            # c_pos +=1
        index += 1
    #     c_all += 1
    # print("possibility of positive review is")
    # print(c_pos/c_all)
    freq_pos = defaultdict(int)
    for word in pos:
        freq_pos[word] += 1
    freq_neg = defaultdict(int)
    for word in neg:
        freq_neg[word] += 1
    n_pos = len(pos)
    n_neg = len(neg)
    v_pos = len(freq_pos)
    v_neg = len(freq_neg)
    #
    # print(n_pos,n_neg)
    # print(v_pos,v_neg)
    # number of frequency of each word in each type counted
    # got P(W|PN)
 
    # for key, value in freq_pos.items():
    #     print("%s : % d" % (key, value))
    ###
    # P(W|PN) = count(W) / n
    # count(W) = number of times W occurs in the documents
    # n = number of total words in the class of documents 
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        possibility_pos = pos_prior
        possibility_neg = 1 - pos_prior
        for word in doc:
            possibility_pos += math.log((freq_pos[word]+ laplace) / (n_pos + laplace*(v_pos+1)))

            possibility_neg += math.log((freq_neg[word]+ laplace) / (n_neg + laplace*(v_neg+1)))
        # to be modified, shoule be 0's and 1's
        if (possibility_pos > possibility_neg):
            yhats.append(1)
        else:
            yhats.append(0)
        print(possibility_pos,possibility_neg)

    return yhats
