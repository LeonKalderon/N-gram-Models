#from sklearn.model_selection import train_test_split
#from nltk import word_tokenize, sent_tokenize
import nltk
import math
import re  #, string, unicodedata
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict


def load_corpus(input_path: str) -> str:
    """
    :param String input_path: gets the path of corpus file and loads it.
    :return String corpus: return the whole corpus
    """
    circlefile = open(input_path, encoding="utf8")
    corpus = circlefile.read()
    circlefile.close()

    # Print a slice of the corpus
    #print('--------CORPUS SAMPLE-----------') #Comment out by GV
    #print(corpus[0:100])

    return corpus


def to_lowercase(words: List[str]) -> List[str]:
    """
    Convert all characters to lowercase from list of tokenized words

    :param List[str] words: A list of words(tokens)
    :return List[str] new_words: the list of parameter words to lower case.
    """
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words: List[str]) -> List[str]:
    """
    Remove punctuation from list of tokenized words

    :param List[str] words: A list of words(tokens)
    :return List[str] new_words: the list of words with all the punctuations removed.
    """
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

    :param List[str] sentence_tokenize: a tokenized sentence
    :param dict mapping: tokens that occur more than some times
    :return List[str] sent_replaced_unk: the replaced sentence
    """

    sent_replaced_unk = [mapping[v] for v in sentence_tokenize]

    return sent_replaced_unk


def tokenize_sentences_and_padding(corpus_sentences, mapping, ngram='bigram', use_padding=True):
    """
    1. Split each sentence into tokens
    2. normalize each sentence (lower case, remove puncts)
    3. Add start/end tokens in cases of bigrams/trigrams when use_padding = True

    :param List[List[str]] corpus_sentences: TODO
    :param Dict mapping: TODO
    :param str ngram: The order of the n-gram model (currently only 2-gram and 3-gram is implemented)
    :param boolean use_padding: TODO
    :return List[List[str]] sentences_tokenized: TODO
    """

    sentences_tokenized = []

    for sent in corpus_sentences:
        sent_tok = nltk.word_tokenize(sent)
        sent_tok = normalize(sent_tok)
        sent_tok = replace(sentence_tokenize=sent_tok, mapping=mapping)
        if use_padding is True:
            if ngram == 'bigram':
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
    bigrams = nltk.bigrams(words)
    bigram_freq = nltk.FreqDist(bigrams)
    vocab_size = len(unigram_freq)
    if model == 'bigram':
        del bigram_freq[('end1', 'start1')]
        ngram_range = 1
    elif model == 'trigram':
        trigrams = nltk.trigrams(words)
        trigram_freq = nltk.FreqDist(trigrams)

        remove = [k for k in trigram_freq.keys() if k[2] in ['start1', 'start2']]
        for k in remove:
            del trigram_freq[k]
        del bigram_freq[('end1', 'start1')]
        del bigram_freq[('start1', 'start2')]
        ngram_range = 2

    for a in range(1, 45, 1):
        alpha = a / 10000
        sum_prob = 0
        trigram_cnt = 0
        for sent in dev_sentences_set:
            for idx in range(ngram_range, len(sent)):
                if model == 'bigram':
                    prob = (bigram_freq[(sent[idx - 1], sent[idx])] + alpha) / (unigram_freq[sent[idx - 1]]
                                                                                + alpha * vocab_size)
                elif model == 'trigram':
                    prob = (trigram_freq[(sent[idx - 2], sent[idx - 1], sent[idx])] + alpha) / (
                                                        bigram_freq[(sent[idx - 2], sent[idx-1])] + alpha * vocab_size)
                sum_prob += math.log2(prob)
                trigram_cnt += 1

        HC = -sum_prob / trigram_cnt
        perpl = math.pow(2, HC)
        results[alpha] = (HC, perpl)

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

# Changed by GV see above function
#def get_min_cross_entropy(res):
#    min_ce = 100000000
#    for key, value in res.items():
#        if value[0] < min_ce:
#            minKey = key
#            min_ce = value[0]
#    return minKey, min_ce


def get_min_cross_entropy(res):
    minKey = min(res, key = res.get)
    min_ce = res[minKey][0]
    min_perplexity = res[minKey][1]
    return minKey, min_ce,min_perplexity

############################################


def get_LM_model(
        unigram_freq_bi: Dict[str, int],
        bigram_freq_bi: Dict[Tuple[str, str], int],
        bigram_freq_tri: Dict[Tuple[str, str], int],
        trigram_freq_tri: Dict[Tuple[str, str, str], int],
        dev_sent_tokenized_trigram: List[List[str]],
        alpha_bigram: float,
        alpha_trigram: float
) -> Dict[float, Tuple[float, float]]:
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

    for l in range(0, 11, 1):
        lamda = l / 10
        for sent in dev_sent_tokenized_trigram:
            for idx in range(2, len(sent)):
                trigram_prob = (trigram_freq_tri[(sent[idx - 2], sent[idx - 1], sent[idx])] + alpha_trigram) / \
                               (bigram_freq_tri[(sent[idx - 2], sent[idx-1])] + alpha_trigram * vocab_size)

                bigram_prob = (bigram_freq_bi[(sent[idx - 1], sent[idx])] + alpha_bigram) / \
                              (unigram_freq_bi[sent[idx - 1]] + alpha_bigram * vocab_size)

                sum_prob += (lamda * math.log2(trigram_prob)) + ((1 - lamda) * math.log2(bigram_prob))
                ngram_cnt += 1

        HC = -sum_prob / ngram_cnt
        perpl = math.pow(2, HC)
        results[lamda] = (HC, perpl)
    return results


def get_random_words(unigrams: List[Dict[Tuple[str], int]], number_of_words: int) -> List[str]:
    """
    :param List[Dict[Tuple(str), int]] unigrams: a list with unigrams
    :param int number_of_words: the number of random unigrams we want to get
    :return List[str] words: a list with random words
    """
    words = []
    size = len(unigrams) - 1
    for i in range(number_of_words):
        words.append(unigrams[random.randint(1, size)])
    return words


def get_random_sentences(sentences: List[List[str]], number_of_sentences: int) -> List[List[str]]:
    """
    :param List[List[str]] sentences: a list with tokenized sentences
    :param int number_of_sentences: the number of random sentences we want to get
    :return List[List[str]] sentence: a list with the random sentences
    """
    sentence = []
    size = len(sentences) - 1
    for i in range(number_of_sentences):
        sentence.append(sentences[random.randint(0, size)])
    return sentence


def calculate_probabilities(
        sentence: List[str],
        unigram_freq_bi: Dict[str, int],
        bigram_freq_bi: Dict[Tuple[str, str], int],
        bigram_freq_tri: Dict[Tuple[str, str], int],
        trigram_freq_tri: Dict[Tuple[str, str, str], int],
        alpha_bigram: float,
        alpha_trigram: float,
        model: str='bigram'
) -> float:

    vocab_size = len(unigram_freq_bi)

    sum_prob = 0
    ngram_cnt = 0

    if model == 'bigram':
        ngram_range = 1

        for idx in range(ngram_range, len(sentence)):
            if model == 'bigram':
                prob = (bigram_freq_bi[(sentence[idx - 1], sentence[idx])] + alpha_bigram) /\
                       (unigram_freq_bi[sentence[idx - 1]] + alpha_bigram * vocab_size)
                sum_prob += math.log2(prob)
                ngram_cnt += 1

    elif model == 'trigram':
        ngram_range = 2
        for idx in range(ngram_range, len(sentence)):
            prob = (trigram_freq_tri[(sentence[idx - 2], sentence[idx - 1], sentence[idx])] + alpha_trigram) / (
                bigram_freq_tri[(sentence[idx - 2], sentence[idx-1])] + alpha_trigram * vocab_size)

            sum_prob += math.log2(prob)
            ngram_cnt += 1

    return -sum_prob