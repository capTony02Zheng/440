"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}

    tag_words_map = dict()
    tag_prevtags_map = dict()
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    N = len(sentences)
    for i in range(N):
        sentence = sentences[i]

        prev_tag = None # store the previous tag
        M = len(sentence)
        for j in range(M):
            word, tag = sentence[j]
            
            # (init) only for start up
            if (j == 1):
                init_prob[tag] += 1

            # (emit)
            emit_prob[tag][word] += 1

            # (trans) access prev tags
            if (prev_tag != None):
                trans_prob[prev_tag][tag] += 1

            # update word count for current tag
            if (tag not in tag_words_map):
                tag_words_map[tag] = list()
            tag_words_map[tag].append(word)

            if (prev_tag != None):
                if (prev_tag not in tag_prevtags_map):
                    tag_prevtags_map[prev_tag] = list()
                tag_prevtags_map[prev_tag].append(tag)

            # update the prev tag
            prev_tag = tag

    # (init)
    total_num = sum(init_prob.values())
    for tag in init_prob.keys():
        init_prob[tag] /= total_num

    # (emit)
    for tag, word_dict in emit_prob.items():
        nt = len(tag_words_map[tag]) # total counts
        vt = len(set(tag_words_map[tag])) # unique counts
        for word in word_dict:
            emit_prob[tag][word] = (emit_prob[tag][word] + emit_epsilon) / (nt + emit_epsilon * (vt + 1))

        # add laplace for unknown
        emit_prob[tag]["UNK"] = emit_epsilon / (nt + emit_epsilon * (vt + 1))

    # (trans)
    for prev_tag, tag_dict in trans_prob.items():
        nt = len(tag_prevtags_map[prev_tag]) # total counts
        vt = len(set(tag_prevtags_map[prev_tag])) # unique counts
        for tag in tag_dict:
            trans_prob[prev_tag][tag] = (trans_prob[prev_tag][tag] + epsilon_for_pt) / (nt + epsilon_for_pt * (vt + 1))

        trans_prob[prev_tag]["UNK"] = epsilon_for_pt / (nt + epsilon_for_pt * (vt + 1))
    trans_prob["END"]["UNK"] = epsilon_for_pt / (nt + epsilon_for_pt * (vt + 1))

    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.

    for tag in prev_prob.keys():
        curr_emit_prob = emit_prob[tag][word]
        if (curr_emit_prob == 0):
            curr_emit_prob = emit_prob[tag]["UNK"]

        best_prevtag_prob = float("-inf")
        best_prevtag = None
        for prev_tag in prev_prob.keys():
            curr_total_prob = 0
            if (i != 0):
                curr_trans_prob = trans_prob[prev_tag][tag]
                if (curr_trans_prob == 0):
                    curr_trans_prob = trans_prob[prev_tag]["UNK"]
                curr_total_prob = prev_prob[prev_tag] + log(curr_emit_prob) + log(curr_trans_prob)

            else:
                curr_total_prob = prev_prob[prev_tag] + log(curr_emit_prob)

            # find best prev_tag that connect curr_tag
            if (curr_total_prob > best_prevtag_prob):
                best_prevtag_prob = curr_total_prob
                best_prevtag = prev_tag

        # update the map
        log_prob[tag] = best_prevtag_prob
        predict_tag_seq[tag] = prev_predict_tag_seq[best_prevtag] + [(word, tag)]

    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.

        # find the best tags
        max_log_prob = float("-inf")
        best_tag = None
        for tag in log_prob.keys():
            curr_log_prob = log_prob[tag]
            if (curr_log_prob > max_log_prob):
                max_log_prob = curr_log_prob
                best_tag = tag
        predicts.append(predict_tag_seq[best_tag])
        
    return predicts