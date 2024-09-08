"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5
hapax_prefix_suffix_map = defaultdict(lambda: defaultdict(lambda: 0))

def suffix_prefix_helper(word, tag):
	weight = 1e-11 # tune the weight

	# 1-
	if (len(word) >= 1 and word[0].isdigit()):
		hapax_prefix_suffix_map["P-NUM"][tag] += weight

	# -1
	if (len(word) >= 1 and word[-1].isdigit()):
		hapax_prefix_suffix_map["S-NUM"][tag] += weight

	# -s 
	if (len(word) >= 1 and word[-1] == 's'):
		hapax_prefix_suffix_map["S-S"][tag] += weight

	# -er 
	if (len(word) >= 2 and word[-2:] == 'er'):
		hapax_prefix_suffix_map["S-ER"][tag] += weight

	# -e
	if (len(word) >= 1 and word[-1] == 'e'):
		hapax_prefix_suffix_map["S-E"][tag] += weight

	# -ed
	if (len(word) >= 2 and word[-2:] == 'ed'):
		hapax_prefix_suffix_map["S-ED"][tag] += weight

	# -ess
	if (len(word) >= 3 and word[-3:] == 'ess'):
		hapax_prefix_suffix_map["S-ESS"][tag] += weight

	# -ing
	if (len(word) >= 3 and word[-3:] == 'ing'):
		hapax_prefix_suffix_map["S-ING"][tag] += weight

	# -ate
	if (len(word) >= 3 and word[-3:] == 'ate'):
		hapax_prefix_suffix_map["S-ATE"][tag] += weight

	# -ent
	if (len(word) >= 3 and word[-3:] == 'ent'):
		hapax_prefix_suffix_map["S-ENT"][tag] += weight

	# -ly
	if (len(word) >= 2 and word[-2:] == 'ly'):
		hapax_prefix_suffix_map["S-LY"][tag] += weight

	# -en
	if (len(word) >= 2 and word[-2:] == 'en'):
		hapax_prefix_suffix_map["S-EN"][tag] += weight

	# -al
	if (len(word) >= 2 and word[-2:] == 'al'):
		hapax_prefix_suffix_map["S-AL"][tag] += weight

	# -ness
	if (len(word) >= 4 and word[-4:] == 'ness'):
		hapax_prefix_suffix_map["S-NESS"][tag] += weight

	# a-
	if (len(word) >= 1 and word[0] == 'a'):
		hapax_prefix_suffix_map["P-A"][tag] += weight

	# al-
	if (len(word) >= 2 and word[:2] == 'al'):
		hapax_prefix_suffix_map["P-AL"][tag] += weight

	# th-
	if (len(word) >= 2 and word[:2] == 'th'):
		hapax_prefix_suffix_map["P-TH"][tag] += weight

	# re-
	if (len(word) >= 2 and word[:2] == 're'):
		hapax_prefix_suffix_map["P-RE"][tag] += weight

	# un-
	if (len(word) >= 2 and word[:2] == 'un'):
		hapax_prefix_suffix_map["P-UN"][tag] += weight

	# dis-
	if (len(word) >= 3 and word[:3] == 'dis'):
		hapax_prefix_suffix_map["P-DIS"][tag] += weight

	# mis-
	if (len(word) >= 3 and word[:3] == 'mis'):
		hapax_prefix_suffix_map["P-MIS"][tag] += weight

	# over-
	if (len(word) >= 4 and word[:4] == 'over'):
		hapax_prefix_suffix_map["P-OVER"][tag] += weight

	# '
	if (len(word) >= 2 and '\'' in word):
		hapax_prefix_suffix_map["S-DOT"][tag] += weight

	# -
	if (len(word) >= 2 and "-" in word):
		hapax_prefix_suffix_map["S-BAR"][tag] += weight

	# ,
	if (len(word) >= 2 and "," in word):
		hapax_prefix_suffix_map["S-COMMA"][tag] += weight

	# .
	if (len(word) >= 2 and "." in word):
		hapax_prefix_suffix_map["S-PERIOD"][tag] += weight

	# /
	if (len(word) >= 2 and "/" in word):
		hapax_prefix_suffix_map["S-SLASH"][tag] += weight

	# /
	if (len(word) >= 2 and ":" in word):
		hapax_prefix_suffix_map["S-COLON"][tag] += weight

	# ?
	if (len(word) >= 2 and "?" in word):
		hapax_prefix_suffix_map["S-QUES"][tag] += weight

def find_prefix_suffix_prob(word, tag):
	max_prob = float("-inf")

	# 1-
	if (len(word) >= 1 and word[0].isdigit()):
		if (hapax_prefix_suffix_map["P-NUM"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["P-NUM"][tag]

	# -1
	if (len(word) >= 1 and word[-1].isdigit()):
		if (hapax_prefix_suffix_map["S-NUM"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-NUM"][tag]

	# -s 
	if (len(word) >= 1 and word[-1] == 's'):
		if (hapax_prefix_suffix_map["S-S"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-S"][tag]

	# -er
	if (len(word) >= 2 and word[-2:] == 'er'):
		if (hapax_prefix_suffix_map["S-ER"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-ER"][tag]

	# -e
	if (len(word) >= 1 and word[-1] == 'e'):
		if (hapax_prefix_suffix_map["S-E"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-E"][tag]

	# -ed
	if (len(word) >= 2 and word[-2:] == 'ed'):
		if (hapax_prefix_suffix_map["S-ED"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-ED"][tag]

	# -ess
	if (len(word) >= 3 and word[-3:] == 'ess'):
		if (hapax_prefix_suffix_map["S-ESS"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-ESS"][tag]

	# -ing
	if (len(word) >= 3 and word[-3:] == 'ing'):
		if (hapax_prefix_suffix_map["S-ING"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-ING"][tag]

	# -ate
	if (len(word) >= 3 and word[-3:] == 'ate'):
		if (hapax_prefix_suffix_map["S-ATE"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-ATE"][tag]

	# -ent
	if (len(word) >= 3 and word[-3:] == 'ent'):
		if (hapax_prefix_suffix_map["S-ENT"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-ENT"][tag]

	# -ly
	if (len(word) >= 2 and word[-2:] == 'ly'):
		if (hapax_prefix_suffix_map["S-LY"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-LY"][tag]

	# -en
	if (len(word) >= 2 and word[-2:] == 'en'):
		if (hapax_prefix_suffix_map["S-EN"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-EN"][tag]

	# -al
	if (len(word) >= 2 and word[-2:] == 'al'):
		if (hapax_prefix_suffix_map["S-AL"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-AL"][tag]

	# -ness
	if (len(word) >= 4 and word[-4:] == 'ness'):
		if (hapax_prefix_suffix_map["S-NESS"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-NESS"][tag]

	# a-
	if (len(word) >= 1 and word[0] == 'a'):
		if (hapax_prefix_suffix_map["P-A"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["P-A"][tag]

	# al-
	if (len(word) >= 2 and word[:2] == 'al'):
		if (hapax_prefix_suffix_map["P-AL"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["P-AL"][tag]

	# th
	if (len(word) >= 2 and word[:2] == 'th'):
		if (hapax_prefix_suffix_map["P-TH"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["P-TH"][tag]

	# re-
	if (len(word) >= 2 and word[:2] == 're'):
		if (hapax_prefix_suffix_map["P-RE"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["P-RE"][tag]

	# dis-
	if (len(word) >= 3 and word[:3] == 'dis'):
		if (hapax_prefix_suffix_map["P-DIS"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["P-DIS"][tag]

	# mis-
	if (len(word) >= 3 and word[:3] == 'mis'):
		if (hapax_prefix_suffix_map["P-MIS"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["P-MIS"][tag]

	# over-
	if (len(word) >= 4 and word[:4] == 'over'):
		if (hapax_prefix_suffix_map["P-OVER"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["P-OVER"][tag]

	# un-
	if (len(word) >= 2 and word[:2] == 'un'):
		if (hapax_prefix_suffix_map["P-UN"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["P-UN"][tag]

	# '
	if ('\'' in word):
		if (hapax_prefix_suffix_map["S-DOT"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-DOT"][tag]

	# -
	if ('-' in word):
		if (hapax_prefix_suffix_map["S-BAR"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-BAR"][tag]

	# ,
	if (',' in word):
		if (hapax_prefix_suffix_map["S-COMMA"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-COMMA"][tag]

	# .
	if ('.' in word):
		if (hapax_prefix_suffix_map["S-PERIOD"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-PERIOD"][tag]

	# /
	if ('/' in word):
		if (hapax_prefix_suffix_map["S-SLASH"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-SLASH"][tag]

	# :
	if (':' in word):
		if (hapax_prefix_suffix_map["S-COLON"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-COLON"][tag]

	# ?
	if ('?' in word):
		if (hapax_prefix_suffix_map["S-QUES"][tag] > max_prob):
			max_prob = hapax_prefix_suffix_map["S-QUES"][tag]

	if (max_prob == float("-inf")):
		return 0

	return max_prob


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
	hapax_wordpair_map = dict()
	
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

			# hapax
			if (word not in hapax_wordpair_map):
				hapax_wordpair_map[word] = (1, tag)
			else:
				hapax_wordpair_map[word] = (hapax_wordpair_map[word][0] + 1, tag)

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

	# (emit) hapax
	hapax_tagoccur_map = dict()
	for word in hapax_wordpair_map.keys():
		occurence, tag = hapax_wordpair_map[word]
		if (occurence == 1):
			if (tag not in hapax_tagoccur_map):
				hapax_tagoccur_map[tag] = 0
			hapax_tagoccur_map[tag] += 1

			# prefix & suffix
			suffix_prefix_helper(word, tag)
			

	# convert hapax count to prob
	hapax_tagoccur_sum = sum(hapax_tagoccur_map.values())
	for tag in hapax_tagoccur_map.keys():
		hapax_tagoccur_map[tag] /= hapax_tagoccur_sum
	
	# (emit) actual
	for tag, word_dict in emit_prob.items():
		nt = len(tag_words_map[tag]) # total counts
		vt = len(set(tag_words_map[tag])) # unique counts

		hapax_emit_epsilon = 0.0001 * emit_epsilon
		if (tag in hapax_tagoccur_map):
			hapax_emit_epsilon = hapax_tagoccur_map[tag] * emit_epsilon

		for word in word_dict:
			emit_prob[tag][word] = (emit_prob[tag][word] + hapax_emit_epsilon) / (nt + hapax_emit_epsilon * (vt + 1))

		emit_prob[tag]["UNK"] = hapax_emit_epsilon / (nt + hapax_emit_epsilon * (vt + 1))

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
			curr_emit_prob = find_prefix_suffix_prob(word, tag)

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

def viterbi_3(train, test):
	'''
	input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
			test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
	output: list of sentences, each sentence is a list of (word,tag) pairs.
			E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
	'''
	# TODO: find useful suffixes and prefixes in all the hapax words, then give it a custom tag	
	init_prob, emit_prob, trans_prob = training(train)
	
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

	# reset global variable
	hapax_prefix_suffix_map = defaultdict(lambda: defaultdict(lambda: 0))
		
	return predicts