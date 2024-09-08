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
    init_tag_counter = Counter()
    tag_tag_count_dict = {}
    tagWordDict = {}
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    # uniqueWordCount = Counter()
    # tagWordCount = Counter()
    for sentence in sentences:
        prevTag = None
        init_tag_counter.update([sentence[0][1]])
        for word,tag in sentence:

            if tag in tagWordDict:
                tagWordDict[tag].update([word]) # counter for each word given a tag
            else:
                tagWordDict[tag] = Counter([word])

            if prevTag is None:
                prevTag = tag
            else:
                if prevTag in tag_tag_count_dict:
                    tag_tag_count_dict[prevTag].update([tag])
                else:
                    tag_tag_count_dict[prevTag] = Counter([tag])
                prevTag = tag

    # produce transition prob
    tagKeys = list(tag_tag_count_dict.keys())
    tagKeys.append("END")
    tag_tag_count_dict["END"] = Counter()
    # print(tagKeys)
    # exit(2)
    for prevTag, tagCounter in tag_tag_count_dict.items():
        total_count = sum(tagCounter.values())
        tagDict = {}
        num_of_unseen_next_tag = len(tagKeys) - len(tagCounter)
        denom = total_count + epsilon_for_pt * (len(tagKeys))
        for nextTag,count in tagCounter.items():
            tagDict[nextTag] = (count+epsilon_for_pt)/denom
        for nextTag in tagKeys:
            if nextTag not in tagDict:
                tagDict[nextTag] = epsilon_for_pt/denom
                # if (tagDict[nextTag]) == 0:
                #     exit(11)
        trans_prob[prevTag] = tagDict

    # produce the emit prob
    for tag, word_counter in tagWordDict.items():
        total_count = sum(word_counter.values())
        num_unique_words_for_tag = len(word_counter)
        denom = total_count + emit_epsilon * (num_unique_words_for_tag + 1)
        tagDict = {}
        # add UNKNOWN
        tagDict["UNKNOWN"] = emit_epsilon / denom
        for word, count in word_counter.items():
            # print(word)
            tagDict[word] = (count+emit_epsilon)/denom
        emit_prob[tag] = tagDict
    total_init_tag_count = sum(init_tag_counter.values())
    for tag, count in init_tag_counter.items():
        # print("tag",tag," count ",count)
        init_prob[tag] = count / total_init_tag_count

    # print("init_prob", init_prob['START'])
    # print(emit_prob["END"])
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
    if i != 0:
        # Start
        # print("prev prob ", prev_prob)
        for tag in prev_prob:
            # generate log prob for each tag
            first = True
            maxProb = 0
            max_prob_prev_tag = ""
            # print("CURRENT TAG IS ",tag)
            for prevTag, logProb in prev_prob.items():
                if word in emit_prob[tag]:
                    # I know this word
                    new_prob = prev_prob[prevTag] + log(trans_prob[prevTag][tag]) + log(emit_prob[tag][word])
                else:
                    # use UNKNOWN
                    # print("trans prob ", trans_prob[prevTag][tag])
                    # print("emit prob ", emit_prob[tag]["UNKNOWN"])
                    # print("prev tag ",prevTag, " next tag ", tag)
                    new_prob = prev_prob[prevTag] + log(trans_prob[prevTag][tag]) + log(emit_prob[tag]["UNKNOWN"])
                if new_prob > maxProb or first:
                    first = False
                    max_prob_prev_tag = prevTag
                    maxProb = new_prob
            log_prob[tag] = maxProb
            # print("max_prob ", max_prob_prev_tag)

            predict_tag_seq[tag] = prev_predict_tag_seq[max_prob_prev_tag] + [(word,tag)]
    else:
        # print("testst", prev_prob)
        for tag in prev_prob:
            # print("tag ", tag, " emit tag " ,emit_prob[tag])
            if word in emit_prob[tag]:
                log_prob[tag] = (prev_prob[tag]) +  log(emit_prob[tag][word])# emit prob +
            else:
                log_prob[tag] = (prev_prob[tag]) + log(emit_prob[tag]["UNKNOWN"])
            predict_tag_seq[tag] = [(word,tag)]

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
                # print("tag", t, " initprob ", init_prob[t])
                log_prob[t] = log(init_prob[t])
            else:
                # print("tag ", t)
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        highest_prob = log_prob["END"]
        highest_prob_tag = "END"
        for tag,prob in log_prob.items():
            if prob > highest_prob:
                highest_prob = prob
                highest_prob_tag = tag
        # print("Highest prob last ", highest_prob_tag)
        predicts.append(predict_tag_seq[highest_prob_tag])
    # print(predicts)
    return predicts




