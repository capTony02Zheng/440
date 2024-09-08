"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
	'''
	input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
			test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
	output: list of sentences, each sentence is a list of (word,tag) pairs.
			E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
	'''
	wordtag_map = dict()
	word_taglist_map = dict()
	tag_occurence_map = dict()

	# add training data into the map
	N = len(train)
	for i in range(N):
		sentence = train[i]
		
		M = len(sentence)
		for j in range(1, M - 1):
			wordtag_pair = sentence[j]

			# if pair exists in wordtag_map
			if (wordtag_pair in wordtag_map):
				wordtag_map[wordtag_pair] += 1
			else:
				wordtag_map[wordtag_pair] = 1


			# keep tracking each word's tags
			word = wordtag_pair[0]
			tag = wordtag_pair[1]
			if (word in word_taglist_map):
				word_tagslist = word_taglist_map[word]

				if (tag not in word_tagslist):
					word_tagslist.append(tag)
			else:
				word_tagslist = list()
				word_tagslist.append(tag)
				word_taglist_map[word] = word_tagslist


			# tracking each tag's occurence time
			if (tag in tag_occurence_map):
				tag_occurence_map[tag] += 1
			else:
				tag_occurence_map[tag] = 1

		
	# most common tag
	most_common_occurence = 0
	most_common_tag = None
	for key in tag_occurence_map.keys():
		if (tag_occurence_map[key] > most_common_occurence):
			most_common_occurence = tag_occurence_map[key]
			most_common_tag = key


	# predicting testing data
	result = []

	N = len(test)
	for i in range(N):
		sentence = test[i]
		M = len(sentence)

		# init the sentence result that starts with the START tag
		sentence_result = []
		sentence_result.append(("START", "START"))

		for j in range(1, M - 1):
			word = sentence[j]

			if word in word_taglist_map:
				word_tagslist = word_taglist_map[word]
				max_occur = 0
				max_occur_tag = None

				# Loop through all the tags for the current word
				for tag in word_tagslist:
					wordtag_pair = (word, tag)
					num_occur = wordtag_map.get(wordtag_pair, 0)
					if num_occur > max_occur:
						max_occur = num_occur
						max_occur_tag = tag

				sentence_result.append((word, max_occur_tag))


			# If the word is unseen
			else:
				sentence_result.append((word, most_common_tag))

		# add the END tag as well
		sentence_result.append(("END", "END"))

		# append sentence result to final result
		result.append(sentence_result)
		
	return result