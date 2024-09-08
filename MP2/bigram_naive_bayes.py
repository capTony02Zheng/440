# bigram_naive_bayes.py
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
from tqdm import tqdm
from collections import defaultdict
from collections import Counter


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.015, bigram_laplace=0.005, bigram_lambda=0.75, pos_prior=0.8, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    # training set is a list of lists of words (each list of words contains all the words in one review)
    # labels (positive or negative review) for the training set
    # list of 0's (Negative) and 1's (Positive)
    # ##
    index = 0;
    freq_pos = defaultdict(int)
    freq_neg = defaultdict(int)
    freq_pospair = defaultdict(int)
    freq_negpair = defaultdict(int)
    n_pos = 0
    n_neg = 0
    n_pospair = 0
    n_negpair = 0
    for doc in train_set:
        if (train_labels[index] == 0):
            n_neg += len(doc)
            n_negpair += (len(doc) - 1)
            for i in range(len(doc)):
                #deal with word
                word = doc[i]
                freq_neg[word] += 1
                #
                #deal with pair
                if (i == len(doc) - 1):
                    break
                pairs = (doc[i]+" "+doc[i+1])
                freq_negpair[pairs] += 1
            # print("test break")
        else:
            n_pos += len(doc)
            n_pospair += (len(doc) - 1)
            for i in range(len(doc)):
                #deal with word
                word = doc[i]
                freq_pos[word] += 1
                #
                #deal with pair
                if (i == len(doc) - 1):
                    break
                pairs = (doc[i]+" "+doc[i+1])
                freq_pospair[pairs] += 1
        index += 1
    v_pos = len(freq_pos)
    v_neg = len(freq_neg)
    v_pospair = len(freq_pospair)
    v_negpair = len(freq_negpair)
    # print(n_pos,n_neg,n_pospair,n_negpair)
    # print(v_pos,v_neg,v_pospair,v_negpair)
    #
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
        possibility_pospair = pos_prior
        possibility_negpair = 1 - pos_prior
        for i in range(len(doc)):
            #word part
            word = doc[i]
            possibility_pos += math.log((freq_pos[word]+ unigram_laplace) / (n_pos + unigram_laplace*(v_pos+1)))
            possibility_neg += math.log((freq_neg[word]+ unigram_laplace) / (n_neg + unigram_laplace*(v_neg+1)))

            # pair part
            if (i == len(doc) - 1):
                break
            pairs = (doc[i]+" "+doc[i+1])
            possibility_pospair += math.log((freq_pospair[pairs]+ bigram_laplace) / (n_pospair + bigram_laplace*(v_pospair+1)))
            possibility_negpair += math.log((freq_negpair[pairs]+ bigram_laplace) / (n_negpair + bigram_laplace*(v_negpair+1)))
        # print(possibility_pos,possibility_neg)
        # print(possibility_pospair,possibility_negpair)
        possibility_pos_weightedall = (1 - bigram_lambda) * possibility_pos + bigram_lambda * possibility_pospair
        possibility_neg_weightedall = (1 - bigram_lambda) * possibility_neg + bigram_lambda * possibility_negpair
        # print(possibility_pos_weightedall,possibility_neg_weightedall)
        if (possibility_pos_weightedall > possibility_neg_weightedall): # to be modified!!!
            yhats.append(1)
        else:
            yhats.append(0)
        # print(possibility_pos_weightedall,possibility_neg_weightedall)
    #python3 mp2.py --laplace 0.005 --bigram_laplace 0.001 --bigram_lambda 0.52 --pos_prior 0.95
    return yhats



