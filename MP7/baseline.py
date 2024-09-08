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
        train_sen_len = len(train)
        dict_wordtag_freq = {}
        dict_tag_freq = {}
        tagset = set()
        for i in range(train_sen_len):
                sentence = train[i]
                for pair in sentence:
                        if pair not in dict_wordtag_freq:
                                dict_wordtag_freq[pair] = 1
                        else:
                                dict_wordtag_freq[pair] += 1

                        if pair[1] not in dict_tag_freq:
                                dict_tag_freq[pair[1]] = 1
                        else:
                                dict_tag_freq[pair[1]] += 1
                        tagset.add(pair[1])
        #### count done
        mfq_tag = max(dict_tag_freq, key=lambda k: dict_tag_freq[k])

        out = []
        test_sen_len = len(test)
        for i in range(test_sen_len):
                sentence = test[i]
                sentence_with_tag = []
                for word in sentence:
                        final_tag = mfq_tag
                        count = 0
                        for tag in tagset:
                                pair = tuple([word,tag])
                                if(pair in dict_wordtag_freq and dict_wordtag_freq[pair] > count):
                                        final_tag = tag
                                        count = dict_wordtag_freq[pair]
                        pair_toadd = tuple([word,final_tag])
                        sentence_with_tag.append(pair_toadd)
                out.append(sentence_with_tag)
        return out