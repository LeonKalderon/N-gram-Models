#Import essential libraries
import nltk
import re
import math
from sklearn.model_selection import train_test_split
import numpy as np
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re, string, unicodedata
#!pip install git+git://github.com/kootenpv/contractions.git
import contractions
#!pip install inflect
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import matplotlib
import matplotlib.pyplot as plt

def load_corpus(input_path):
    """
    * Description: gets the path of corpus file and loads it.
    * Input: path of corpus file to load
    * retutns corpus
    """

    circlefile = open(input_path, encoding="utf8")
    corpus = circlefile.read()
    circlefile.close()
    '''
    corpus = 'NLTK is a leading platform for building Python programs to work with human language data. ' \
             'It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries ' \
             'for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, ' \
             'and an active discussion forum.Thanks to a hands-on guide introducing programming fundamentals alongside topics in computational linguistics, ' \
             'plus comprehensive API documentation, NLTK is suitable for linguists, engineers, students, educators, researchers, and industry users alike. ' \
             'NLTK is available for Windows, Mac OS X, and Linux. Best of all, NLTK is a free, open source, community-driven project.NLTK has been called ' \
             'a wonderful tool for teaching, and working in, computational linguistics using Python, and an amazing library to play with natural language.Natural Language ' \
             'Processing with Python provides a practical introduction to programming for language processing. Written by the creators of NLTK, it guides the reader through ' \
             'the fundamentals of writing Python programs, working with corpora, categorizing text, analyzing linguistic structure, and more.'
    # Print a slice of the corpus
     '''
    print('--------CORPUS SAMPLE-----------')
    print(corpus[0:1000])


    return corpus

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def normalize(words):
    """
    Gets a list of words, cnverts to lowercase and
    removes punctuation
    """
    words = to_lowercase(words)
    words = remove_punctuation(words)

    return words

def get_unigram_freq(corpus):
    """
    Gets the copus, tokenizes it to words, converts to lowercase,
    removes punctuation and calculates the frequency of each word.
    """
    corpus_tokenize = nltk.word_tokenize(corpus)
    corpus_tokenize = normalize(corpus_tokenize)

    return nltk.FreqDist(corpus_tokenize)

def filter_by_freq_words(unigram_freq_corpus, freq = 10):
    """
    Finds the tokens that occur more than a given number of times.
    """
    notunk = {k: v for k, v in unigram_freq_corpus.items() if v >= freq}

    unk1 = list(notunk)
    len(unk1)
    mapping = nltk.defaultdict(lambda: 'UNK')
    for v in unk1:
        mapping[v] = v

    return mapping

def replace(sentence_tokenize, mapping):
    """
    This function takes as input a tokenized sentence and the tokens that occur more than some times
    If the token do exist in mapping variable it replaces it with the token "UNK"
    """
    sent_replaced_UNK = [mapping[v] for v in sentence_tokenize]

    return sent_replaced_UNK

def tokenize_sentences_and_padding(corpus_sentences, mapping, ngram='bigram'):
    """
    1. Split each sentence into tokens
    2. normalize each sentence (lower case, remove puncts)
    3. Add start/end tokens in cases of bigrams/trigrams
    """
    sentences_tokenized = []
    i = 0
    for sent in corpus_sentences:
        sent_tok = nltk.word_tokenize(sent)
        sent_tok = normalize(sent_tok)
        sent_tok = replace(sentence_tokenize=sent_tok, mapping=mapping)
        if (ngram == 'bigram'):
            sent_tok = ['start1'] + sent_tok + ['end1']
        elif ngram == 'trigram':
            sent_tok = ['start1', 'start2'] + sent_tok + ['end1']

        sentences_tokenized.append(sent_tok)

    return sentences_tokenized

def get_cross_entropy_perplexity(model, words, dev_sentences_set):
    """
    Compute corpus cross_entropy
    & perplexity
    """
    results = {}
    unigram_freq = nltk.FreqDist(words)
    vocab_size = len(unigram_freq)
    bigrams = nltk.bigrams(words)
    bigram_freq = nltk.FreqDist(bigrams)

    if model == 'bigram':
        del bigram_freq[('end1', 'start1')]
        ngram_range = 1
    elif model == 'trigram':
        trigrams = nltk.trigrams(words)
        trigram_freq = nltk.FreqDist(trigrams)

        remove = [k for k in trigram_freq.keys() if k[2] in ['start1', 'start2']]
        for k in remove: del trigram_freq[k]
        del bigram_freq[('end1', 'start1')]
        del bigram_freq[('start1', 'start2')]
        ngram_range=2

    for a in range(1, 11, 1):
        alpha = a / 10
        sum_prob = 0
        trigram_cnt = 0
        print(alpha)
        for sent in dev_sentences_set:
            for idx in range(ngram_range, len(sent)):
                if model=='bigram':
                    prob = (bigram_freq[(sent[idx - 1], sent[idx])] + alpha) / (
                    unigram_freq[sent[idx - 1]] + alpha * vocab_size)
                elif model=='trigram':
                    prob = (trigram_freq[(sent[idx - 2], sent[idx - 1], sent[idx])] + alpha) / (
                bigram_freq[(sent[idx - 1], sent[idx])] + alpha * vocab_size)

                sum_prob += math.log2(prob)
                trigram_cnt += 1

        HC = -sum_prob / trigram_cnt
        perpl = math.pow(2, HC)
        results[alpha] = (HC, perpl)

        print("Cross Entropy: {0:.3f}".format(HC))
        print("perplexity: {0:.3f}".format(perpl))

    return results

def plot_perplexity(perplexity_results):
    alpha = []
    perplexity = []
    cross_entropies = []
    for key, value in perplexity_results.items():
        alpha.append(key)
        perplexity.append(value[1])
        cross_entropies.append(value[0])

    fig, ax = plt.subplots()
    ax.plot(alpha, perplexity)
    plt.show()


############################################
def get_LM_model(unigram_freq_bi,bigram_freq_bi,bigram_freq_tri,trigram_freq_tri):
    """
    Compute corpus cross_entropy
    & perplexity for interpoladed bi-gram
    & tri-gram LMs
    """
    results = {}
    # We should fine-tune lamda on a held-out dataset
    vocab_size = len(unigram_freq_bi)
    sum_prob = 0
    ngram_cnt = 0
    alpha = 0.1
    for l in range(0, 11, 1):
        lamda = l / 10
        for sent in dev_sent_tokenized_trigram:
            for idx in range(2, len(sent)):
                trigram_prob = (trigram_freq_tri[(sent[idx - 2], sent[idx - 1], sent[idx])] + alpha) / (
                bigram_freq_tri[(sent[idx - 1], sent[idx])] + alpha * vocab_size)
                bigram_prob = (bigram_freq_bi[(sent[idx - 1], sent[idx])] + alpha) / (
                unigram_freq_bi[sent[idx - 1]] + alpha * vocab_size)

                sum_prob += (lamda * math.log2(trigram_prob)) + ((1 - lamda) * math.log2(bigram_prob))
                ngram_cnt += 1

        HC = -sum_prob / ngram_cnt
        perpl = math.pow(2, HC)
        print("Cross Entropy: {0:.3f}".format(HC))
        print("perplexity: {0:.3f}".format(perpl))
        results[lamda] = (HC, perpl)
    return results

############################################
if __name__=='__main__':
    # Read Coprus
    input_path = r'C:\Users\Georgia.Sarri\Documents\Msc\5th\TextAnalytics\Assignmnets\1st\europarl-v7.el-en.en'
    corpus = load_corpus(input_path)[0:1000]
    #Get unigram freq dictionary
    unigram_freq_corpus = get_unigram_freq(corpus)
    print('Frequency of the : {}'.format(unigram_freq_corpus['the']))

    #Get the mapping for the words we will keep
    mapping = filter_by_freq_words(unigram_freq_corpus)

    #Split to sentences
    sentences_corpus = sent_tokenize(corpus)

    #--------------
    # BIGRAM MODEL:
    #--------------
    #Create unigram/bigram/trigram tokenization
    sentences_tokenized_bigram = tokenize_sentences_and_padding(corpus_sentences=sentences_corpus,  mapping=mapping, ngram='bigram')
    print('----SAMPLE OF TOKENIZED BIGRAMS----')
    print(sentences_tokenized_bigram[0:10])
    #Get train and test sets:
    train_sent_tokenized_bigram, test_and_dev_sent_tokenized_bigram = train_test_split(sentences_tokenized_bigram,
                                                                                       test_size=0.2, random_state=1)
    test_sent_tokenized_bigram, dev_sent_tokenized_bigram = train_test_split(test_and_dev_sent_tokenized_bigram,
                                                                             test_size=0.5, random_state=1)

    print('The training corpus contains {} sentences'.format(len(train_sent_tokenized_bigram)))
    print('The developement corpus contains {} sentences'.format(len(dev_sent_tokenized_bigram)))
    print('The test corpus contains {} sentences'.format(len(test_sent_tokenized_bigram)))

    #Flatten the training tokinized corpus and create unigram and bigram frequencies dictionaries
    sentences_tokenized_bigram_flattened = [val for sublist in train_sent_tokenized_bigram for val in sublist]
    ce_perpl_results = get_cross_entropy_perplexity('bigram', sentences_tokenized_bigram_flattened, dev_sent_tokenized_bigram)
    plot_perplexity(ce_perpl_results)

    #----------------
    # TRIGRAM MODEL:
    #----------------
    #Create unigram/bigram/trigram tokenization
    sentences_tokenized_trigram = tokenize_sentences_and_padding(corpus_sentences=sentences_corpus,  mapping=mapping, ngram='trigram')
    print('----SAMPLE OF TOKENIZED TRIGRAMS----')
    print(sentences_tokenized_trigram[0:10])
    #Get train and test sets:
    train_sent_tokenized_trigram, test_and_dev_sent_tokenized_trigram = train_test_split(sentences_tokenized_trigram,
                                                                                       test_size=0.2, random_state=1)
    test_sent_tokenized_trigram, dev_sent_tokenized_trigram = train_test_split(test_and_dev_sent_tokenized_trigram,
                                                                             test_size=0.5, random_state=1)

    print('The training corpus contains {} sentences'.format(len(train_sent_tokenized_trigram)))
    print('The developement corpus contains {} sentences'.format(len(dev_sent_tokenized_trigram)))
    print('The test corpus contains {} sentences'.format(len(test_sent_tokenized_trigram)))

    #Flatten the training tokinized corpus and create unigram and bigram frequencies dictionaries
    sentences_tokenized_trigram_flattened = [val for sublist in train_sent_tokenized_trigram for val in sublist]
    ce_perpl_results_tri = get_cross_entropy_perplexity('trigram', sentences_tokenized_trigram_flattened, dev_sent_tokenized_trigram)
    plot_perplexity(ce_perpl_results_tri)

    #----------------
    #OPTIONAL MODEL:
    #----------------
    unigram_freq_bi = nltk.FreqDist(sentences_tokenized_bigram_flattened)
    bigrams_bi = nltk.bigrams(sentences_tokenized_bigram_flattened)
    bigram_freq_bi = nltk.FreqDist(bigrams_bi)
    del bigram_freq_bi[('end1', 'start1')]
    unigram_freq_tri = nltk.FreqDist(sentences_tokenized_trigram_flattened)
    bigrams_tri = nltk.bigrams(sentences_tokenized_trigram_flattened)
    trigrams_tri = nltk.trigrams(sentences_tokenized_trigram_flattened)
    bigram_freq_tri = nltk.FreqDist(bigrams_tri)
    trigram_freq_tri = nltk.FreqDist(trigrams_tri)
    remove = [k for k in trigram_freq_tri.keys() if k[2] in ['start1', 'start2']]
    for k in remove: del trigram_freq_tri[k]
    del bigram_freq_tri[('end1', 'start1')]
    del bigram_freq_tri[('start1', 'start2')]
    res = get_LM_model(unigram_freq_bi, bigram_freq_bi, bigram_freq_tri, trigram_freq_tri)
    plot_perplexity(res)