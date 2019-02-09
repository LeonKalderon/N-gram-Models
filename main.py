import nltk
#import re
import math
from sklearn.model_selection import train_test_split
#import numpy as np
#nltk.download('punkt')
#from nltk.tokenize import word_tokenize
#import re, string, unicodedata
#!pip install git+git://github.com/kootenpv/contractions.git
#import contractions
#!pip install inflect
#import inflect
from nltk import word_tokenize, sent_tokenize
#from nltk.corpus import stopwords
#from nltk.stem import LancasterStemmer, WordNetLemmatizer
#import matplotlib
#import matplotlib.pyplot as plt
#import random
import ngramHelpers as ng


def main():
    # Read Coprus
    input_path = r'C:\Users\User\Desktop\MSc Courses\Untitled Folder\europarl-v7.el-en.en'
    corpus1 = ng.load_corpus(input_path)
    corpus = corpus1[0:100000]  # Added by GV

    # Split corpus to sentences
    sentences_corpus = sent_tokenize(corpus)

    # Get train and test sets:
    train_and_dev_sent_tokenized, test_sent_tokenized = train_test_split(
        sentences_corpus,
        test_size=0.2,
        random_state=1
    )

    train_sent_tokenized, dev_sent_tokenized = train_test_split(
        train_and_dev_sent_tokenized,
        test_size=0.2,
        random_state=1
    )

    print('The training corpus contains {} sentences'.format(len(train_and_dev_sent_tokenized)))
    print('The developement corpus contains {} sentences'.format(len(dev_sent_tokenized)))
    print('The test corpus contains {} sentences'.format(len(test_sent_tokenized)))

    # Get unigram freq dictionary of train
    train_corpus = " ".join(train_and_dev_sent_tokenized)
    unigram_freq_train = ng.get_unigram_freq(train_corpus)

    # Get the mapping for the words we will keep (tokens with count >= threshold)
    mapping = ng.filter_by_freq_words(unigram_freq_train)

    test_sent_tokenized_bigram = ng.tokenize_sentences_and_padding(
        corpus_sentences=test_sent_tokenized,
        mapping=mapping,
        ngram='bigram', use_padding=True
    )

    test_sent_tokenized_trigram = ng.tokenize_sentences_and_padding(
        corpus_sentences=test_sent_tokenized,
        mapping=mapping,
        ngram='trigram', use_padding=True
    )

    # --------------
    # BIGRAM MODEL:
    # --------------

    # Create unigram/bigram/trigram tokenization
    train_sent_tokenized_bigram = ng.tokenize_sentences_and_padding(
        corpus_sentences=train_sent_tokenized,
        mapping=mapping,
        ngram='bigram'
    )
    dev_sent_tokenized_bigram = ng.tokenize_sentences_and_padding(
        corpus_sentences=dev_sent_tokenized,
        mapping=mapping,
        ngram='bigram'
    )

    # print('----SAMPLE OF TOKENIZED BIGRAMS----')
    # print(train_sent_tokenized_bigram[0:10])

    # Flatten the training tokinized corpus and create unigram and bigram frequencies dictionaries
    sentences_tokenized_bigram_flattened = [val for sublist in train_sent_tokenized_bigram for val in sublist]

    ce_perpl_results_bigram = ng.get_cross_entropy_perplexity(
        model='bigram',
        words=sentences_tokenized_bigram_flattened,
        dev_sentences_set=dev_sent_tokenized_bigram
    )

    ng.plot_perplexity(ce_perpl_results_bigram)

    alpha_bi, value_bi, value1_bi = ng.get_min_cross_entropy(ce_perpl_results_bigram)
    print('-----BI-GRAMS MODEL-----')
    print('Perplexity {}'.format(value1_bi))
    print('Cross Entropy {}'.format(value_bi))
    print('a smoothing {}'.format(alpha_bi))

    # ----------------
    # TRIGRAM MODEL:
    # ----------------

    # Create unigram/bigram/trigram tokenization
    # Get train and test sets:
    train_sent_tokenized_trigram = ng.tokenize_sentences_and_padding(
        corpus_sentences=train_sent_tokenized,
        mapping=mapping,
        ngram='trigram'
    )

    dev_sent_tokenized_trigram = ng.tokenize_sentences_and_padding(
        corpus_sentences=dev_sent_tokenized,
        mapping=mapping,
        ngram='trigram'
    )

    # print('The training corpus contains {} sentences'.format(len(train_sent_tokenized_trigram)))
    # print('The developement corpus contains {} sentences'.format(len(dev_sent_tokenized_trigram)))

    # print('----SAMPLE OF TOKENIZED TRIGRAMS----')
    # print(train_sent_tokenized_trigram[0:10])

    # Flatten the training tokinized corpus and create unigram and bigram frequencies dictionaries
    sentences_tokenized_trigram_flattened = [val for sublist in train_sent_tokenized_trigram for val in sublist]

    ce_perpl_results_tri = ng.get_cross_entropy_perplexity(
        model='trigram',
        words=sentences_tokenized_trigram_flattened,
        dev_sentences_set=dev_sent_tokenized_trigram
    )

    ng.plot_perplexity(ce_perpl_results_tri)

    alpha_tri, value_tri, value1_tri = ng.get_min_cross_entropy(ce_perpl_results_tri)
    print('-----TRI-GRAMS MODEL-----')
    print('Perplexity {}'.format(value1_tri))
    print('Cross Entropy {}'.format(value_tri))
    print('a smoothing {}'.format(alpha_tri))

    # ----------------
    # OPTIONAL MODEL:
    # ----------------

    unigram_freq_bi = nltk.FreqDist(sentences_tokenized_bigram_flattened)
    bigrams_bi = nltk.bigrams(sentences_tokenized_bigram_flattened)
    bigram_freq_bi = nltk.FreqDist(bigrams_bi)
    del bigram_freq_bi[('end1', 'start1')]

    bigrams_tri = nltk.bigrams(sentences_tokenized_trigram_flattened)
    trigrams_tri = nltk.trigrams(sentences_tokenized_trigram_flattened)
    bigram_freq_tri = nltk.FreqDist(bigrams_tri)
    trigram_freq_tri = nltk.FreqDist(trigrams_tri)
    remove = [k for k in trigram_freq_tri.keys() if k[2] in ['start1', 'start2']]
    for k in remove:
        del trigram_freq_tri[k]

    del bigram_freq_tri[('end1', 'start1')]
    del bigram_freq_tri[('start1', 'start2')]

    res = ng.get_LM_model(
        unigram_freq_bi=unigram_freq_bi,
        bigram_freq_bi=bigram_freq_bi,
        bigram_freq_tri=bigram_freq_tri,
        trigram_freq_tri=trigram_freq_tri,
        dev_sent_tokenized_trigram=dev_sent_tokenized_trigram,
        alpha_bigram=alpha_bi,
        alpha_trigram=alpha_tri
    )

    ng.plot_perplexity(res)

    key_lin, value_lin1, value_lin = ng.get_min_cross_entropy(res)
    print('----LINEAR COMBINATION MODEL-----')
    print('Perplexity {}'.format(value_lin))
    print('Cross Entropy {}'.format(value_lin1))
    print('Lambda {}'.format(key_lin))

    print()
    print("-------- CHECK ON RANDOM SENTENCES ----------")
    prob_bi_sent = 0
    prob_bi_word = 0
    prob_tri_sent = 0
    prob_tri_word = 0

    for i in range(1000):
        random_sentence = ng.get_random_sentences(test_sent_tokenized_bigram, 1)[0]

        random_words = ['start1'] + ng.get_random_words(list(unigram_freq_train.keys()),
                                                        len(random_sentence) - 2) + ['end1']

        tmp_rand_sent_bi = ng.calculate_probabilities(
            model='bigram',
            sentence=random_sentence,
            unigram_freq_bi=unigram_freq_bi,
            bigram_freq_bi=bigram_freq_bi,
            bigram_freq_tri=bigram_freq_tri,
            trigram_freq_tri=trigram_freq_tri,
            alpha_bigram=alpha_bi,
            alpha_trigram=alpha_tri
        )

        prob_bi_sent += tmp_rand_sent_bi

        tmp_rand_words_bi = ng.calculate_probabilities(
            model='bigram',
            sentence=random_words,
            unigram_freq_bi=unigram_freq_bi,
            bigram_freq_bi=bigram_freq_bi,
            bigram_freq_tri=bigram_freq_tri,
            trigram_freq_tri=trigram_freq_tri,
            alpha_bigram=alpha_bi,
            alpha_trigram=alpha_tri
        )

        prob_bi_word += tmp_rand_words_bi

        # Fix padding for trigrams
        random_sentence.insert(1, 'start2')

        tmp_rand_sent_tri = ng.calculate_probabilities(
            model='trigram',
            sentence=random_sentence,
            unigram_freq_bi=unigram_freq_bi,
            bigram_freq_bi=bigram_freq_bi,
            bigram_freq_tri=bigram_freq_tri,
            trigram_freq_tri=trigram_freq_tri,
            alpha_bigram=alpha_bi,
            alpha_trigram=alpha_tri
        )

        prob_tri_sent += tmp_rand_sent_tri

        tmp_rand_words_tri = ng.calculate_probabilities(
            model='trigram',
            sentence=random_words,
            unigram_freq_bi=unigram_freq_bi,
            bigram_freq_bi=bigram_freq_bi,
            bigram_freq_tri=bigram_freq_tri,
            trigram_freq_tri=trigram_freq_tri,
            alpha_bigram=alpha_bi,
            alpha_trigram=alpha_tri
        )

        prob_tri_word += tmp_rand_words_tri

    print('BIGRAM')
    print("Mean probability on test sentences: {}".format(prob_bi_sent / 1000))
    print("Mean probability on random words: {}".format(prob_bi_word / 1000))
    print('TRIGRAM')
    print("Mean probability on test sentences: {}".format(prob_tri_sent / 1000))
    print("Mean probability on random words: {}".format(prob_tri_word / 1000))

    print("------- RESULTS ON TEST SET ---------")

    print("----BI-GRAMS MODEL ON TEST SET-------")
    test_sentences_tokenized_trigram_flattened = [val for sublist in test_sent_tokenized_trigram for val in sublist]
    vocab_size = len(unigram_freq_bi)

    sum_prob = 0
    ngram_cnt = 0

    for idx in range(1, len(test_sentences_tokenized_trigram_flattened)):
        if test_sentences_tokenized_trigram_flattened[idx - 1] not in ['start1']:
            prob = (bigram_freq_bi[(test_sentences_tokenized_trigram_flattened[idx - 1],
                                    test_sentences_tokenized_trigram_flattened[idx])] + alpha_tri) / \
                   (unigram_freq_bi[test_sentences_tokenized_trigram_flattened[idx - 1]] + alpha_bi * vocab_size)

            sum_prob += math.log2(prob)
            ngram_cnt += 1

    HC = -sum_prob / ngram_cnt
    perpl = math.pow(2, HC)
    print("Cross Entropy: {0:.3f}".format(HC))
    print("perplexity: {0:.3f}".format(perpl))

    print("----TRI-GRAMS MODEL ON TEST SET-------")

    sum_prob = 0
    ngram_cnt = 0

    for idx in range(2, len(test_sentences_tokenized_trigram_flattened)):
        if test_sentences_tokenized_trigram_flattened[idx - 1] not in ['start1', 'start2']:
            prob = (trigram_freq_tri[(test_sentences_tokenized_trigram_flattened[idx - 2],
                                      test_sentences_tokenized_trigram_flattened[idx - 1],
                                      test_sentences_tokenized_trigram_flattened[idx])] + alpha_tri) / \
                   (
                       bigram_freq_tri[(test_sentences_tokenized_trigram_flattened[idx - 2],
                                        test_sentences_tokenized_trigram_flattened[idx - 1])] + alpha_tri * vocab_size)

            sum_prob += math.log2(prob)
            ngram_cnt += 1

    HC = -sum_prob / ngram_cnt
    perpl = math.pow(2, HC)
    print("Cross Entropy: {0:.3f}".format(HC))
    print("perplexity: {0:.3f}".format(perpl))


if __name__ == '__main__':
    main()
