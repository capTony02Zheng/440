"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

import math
from collections import defaultdict, Counter
from math import log

hapax_pr_sf = defaultdict(lambda: defaultdict(lambda: 0))

def hapax_pr_sf_constructor(word,tag):
        weight = 1e-10
        suf = ["-ey","-ish","-ments","-.","-ss","-s'","-?","-\'","-/","-ed", "-ing", "-ly", "-able", "-ment", "-tion", "-ness", "-er", "-est", "-y", "-ful", "-less", "-ance", "-ive", "-ize", "-ous", "-al", "-ship", "-ize", "-ed", "-est", "-es", "-er", "-ity", "-ize", "-ly", "-ful", "-ous", "-ance", "-ing", "-ist", "-ize", "-age", "-ed", "-er", "-ive", "-ment", "-al", "-ence", "-dom","-'s","-.","-,","-:", "-;", "--","-''"]
        pre = ["a-","self-","**-","$-","un-", "re-", "in-", "dis-", "pre-", "mis-", "over-", "sub-", "inter-", "super-", "anti-", "mid-", "semi-", "under-", "trans-", "co-", "non-", "micro-", "macro-", "auto-", "semi-", "mono-", "bi-", "tri-", "multi-", "anti-", "post-", "pre-", "sub-", "hyper-", "ultra-", "ex-", "semi-", "pro-", "tele-", "eco-", "geo-", "poly-", "quad-", "uni-"]
        mid = ["-ey-","---","-,-","-/-","-.-","-:-"]
        # 1-
        if (len(word) >= 1 and word[0].isdigit()):
                hapax_pr_sf["P-NUM"][tag] += weight
        
        # -1
        if (len(word) >= 1 and word[-1].isdigit()):
                hapax_pr_sf["S-NUM"][tag] += weight
        ### to do
        for s in suf:
              if word.endswith(s[1:]):
                    hapax_pr_sf[s][tag] += weight
        for p in pre:
             if word.startswith(p[:-1]):
                  hapax_pr_sf[p][tag] += weight
        for m in mid:
              if m[1:-1] in word:
                #     print(word,m)
                    hapax_pr_sf[m][tag] += weight

        ###



def hapax_pr_sf_finder(word, tag):
    max_probp = 0
    max_probs = 0
    max_probm = 0

    if isinstance(word, str) and len(word) >= 1:
        # Handle prefixes
        if word[0].isdigit() and hapax_pr_sf["P-NUM"][tag] > max_probp:
            max_probp = hapax_pr_sf["P-NUM"][tag]

        # Handle suffixes
        if word[-1].isdigit() and hapax_pr_sf["S-NUM"][tag] > max_probs:
            max_probs = hapax_pr_sf["S-NUM"][tag]

        # Check other prefixes and suffixes
        suf = ["-ey","-ish","-ments","-.","-ss","-s'","-?","-\'","-/","-ed", "-ing", "-ly", "-able", "-ment", "-tion", "-ness", "-er", "-est", "-y", "-ful", "-less", "-ance", "-ive", "-ize", "-ous", "-al", "-ship", "-ize", "-ed", "-est", "-es", "-er", "-ity", "-ize", "-ly", "-ful", "-ous", "-ance", "-ing", "-ist", "-ize", "-age", "-ed", "-er", "-ive", "-ment", "-al", "-ence", "-dom","-'s","-.","-,","-:", "-;", "--","-''"]
        pre = ["a-","self-","**-","$-","un-", "re-", "in-", "dis-", "pre-", "mis-", "over-", "sub-", "inter-", "super-", "anti-", "mid-", "semi-", "under-", "trans-", "co-", "non-", "micro-", "macro-", "auto-", "semi-", "mono-", "bi-", "tri-", "multi-", "anti-", "post-", "pre-", "sub-", "hyper-", "ultra-", "ex-", "semi-", "pro-", "tele-", "eco-", "geo-", "poly-", "quad-", "uni-"]
        mid = ["-ey-","---","-,-","-/-","-.-","-:-"]
        
        for s in suf:
                if word.endswith(s[1:]) and hapax_pr_sf[s][tag] > max_probp:
                        max_probp = hapax_pr_sf[s][tag]
        for p in pre:
                if word.startswith(p[:-1]) and hapax_pr_sf[p][tag] > max_probs:
                        max_probs = hapax_pr_sf[p][tag]
        for m in mid:
                if m[1:-1] in word and hapax_pr_sf[m][tag] > max_probm:
                        max_probm = hapax_pr_sf[m][tag]

    return sum([max_probp,max_probs,max_probm])

def training(sentences):
        """
        Computes initial tags, emission words and transition tag-to-tag probabilities
        :param sentences:
        :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
        """
        init_prob = defaultdict(lambda: 0) # {init tag: #}
        emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
        trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
        hapax_pr_sf = defaultdict(lambda: defaultdict(lambda: 0))
        
        # TODO: (I)
        # Input the training set, output the formatted probabilities according to data statistics.
        tag_word = {}
        tag_prevtag = {}
        word_freq_tag = {}
        for sentence in sentences:
                prev_tag = None  # Special start tag
                init_prob[sentence[0][1]] += 1  # Count the first tag in the sentence as an initial tag
                for word, tag in sentence:
                        emit_prob[tag][word] += 1  # Count the word being emitted from the tag
                        if (prev_tag != None):
                                trans_prob[prev_tag][tag] += 1  # Count the transition from prev_tag to tag
                        prev_tag = tag
                        if tag not in tag_word:
                                tag_word[tag] = []
                        tag_word[tag].append(word)

                        if prev_tag != None and prev_tag not in tag_prevtag:
                                tag_prevtag[prev_tag] = []
                        if prev_tag != None:
                                tag_prevtag[prev_tag].append(tag)

                        if word not in word_freq_tag:
                              word_freq_tag[word] = (1,tag)
                        else:
                              word_freq_tag[word] = (2,tag)


                              
        
        hapax_tag_freq = {}
        for word in word_freq_tag.keys():
                if(word_freq_tag[word][0] == 1):
                        tag = word_freq_tag[word][1]
                        if tag not in hapax_tag_freq:
                                hapax_tag_freq[tag] = 0
                        hapax_tag_freq[tag] += 1
                        hapax_pr_sf_constructor(word,tag)
                        # with open('hapax2.txt', 'a') as file:
                        #         file.write(word+" "+tag)
                        #         file.write("\n")

        
        
        total_hapax_tags = sum(hapax_tag_freq.values())
        for k in hapax_tag_freq.keys():
              hapax_tag_freq[k] /= total_hapax_tags


        total_initial_tags = sum(init_prob.values())

        for tag in init_prob:
                init_prob[tag] /= total_initial_tags

        emit_epsilon = 1e-5
        for tag, word_dict in emit_prob.items():
                nt = len(tag_word[tag]) # total counts
                vt = len(set(tag_word[tag])) # unique counts

                
                brand_new_epsilon = 1e-9
                if tag in hapax_tag_freq.keys():
                      brand_new_epsilon = hapax_tag_freq[tag] * emit_epsilon

                for word in word_dict:
                        emit_prob[tag][word] = (emit_prob[tag][word] + brand_new_epsilon) / (nt + brand_new_epsilon * (vt + 1))

                # add laplace for unknown
                emit_prob[tag]["UNK"] = brand_new_epsilon / (nt + brand_new_epsilon * (vt + 1))

        for prev_tag, tag_dict in trans_prob.items():
                nt = len(tag_prevtag[prev_tag]) # total counts
                vt = len(set(tag_prevtag[prev_tag])) # unique counts
                for tag in tag_dict:
                        trans_prob[prev_tag][tag] = (trans_prob[prev_tag][tag] + emit_epsilon) / (nt + emit_epsilon * (vt + 1))

                trans_prob[prev_tag]["UNK"] = emit_epsilon / (nt + emit_epsilon * (vt + 1))
        trans_prob["END"]["UNK"] = emit_epsilon / (nt + emit_epsilon * (vt + 1))

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
    tags = emit_prob.keys()
    for tag in tags:
        emit_pr = emit_prob[tag][word]
        if (emit_pr == 0):
            emit_pr = hapax_pr_sf_finder(word,tag)
            if emit_pr == 0:
                emit_pr = emit_prob[tag]["UNK"]
                # print("shit is:"+word+" "+tag)
        #     with open('hapax.txt', 'a') as file:
        #         file.write(word)
        #         file.write("\n")
                
        best_prevtag_prob = -math.inf
        best_prevtag = None

        for prev_tag in tags:
            tt_prob = 0
            if (i != 0):
                trans_pr = trans_prob[prev_tag][tag]
                if (trans_pr == 0):
                    trans_pr = trans_prob[prev_tag]["UNK"]
                tt_prob = prev_prob[prev_tag] + log(emit_pr) + log(trans_pr)
            else:
                tt_prob = prev_prob[prev_tag] + log(emit_pr)

            if (tt_prob > best_prevtag_prob):
                best_prevtag = prev_tag
                best_prevtag_prob = tt_prob

        log_prob[tag] = best_prevtag_prob
        predict_tag_seq[tag] = prev_predict_tag_seq[best_prevtag] + [(word, tag)]
    
    return log_prob, predict_tag_seq

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = training(train)
    
    predicts = []
    epsilon_for_pt = 1e-5
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
        max_log_prob = float("-inf")
        best_tag = None
        for tag in log_prob.keys():
            curr_log_prob = log_prob[tag]
            if (curr_log_prob > max_log_prob):
                max_log_prob = curr_log_prob
                best_tag = tag
        predicts.append(predict_tag_seq[best_tag])
        
    return predicts