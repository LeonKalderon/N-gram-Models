from sklearn.model_selection import train_test_split
from nltk import word_tokenize, sent_tokenize
import nltk
import math
import re, string, unicodedata
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict
import numpy as np
import time
from scipy import sparse
import pandas as pd
'''
We implemented 4 ways (evolving each time) for calculating the KN smoothing rule.
The first one was very slow and was only used for 1.000.000 out of 191.000.000 of the corpus.
It took about an hour and the execution time was increasing exponentially so it would be impossible to do for
the whole corpus.

The second implementation was based on numpy approach and it required around 12 hours for 52% of the corpus
but it would still be demanding for the whole corpus.

The 3rd attempt was based on numpy but using sparse techniques. This approach resulted 
in about 1 hour running for the whole corpus so this one was used for the assignment.

The 4th attempt was based on the code that nltk uses for trigrams modified for bigrams. We
get the same results with the previous models

The other 2 models are also include at the end for the file. If you need to test them
you should reduce the size of the corpus as the take a significant amount of time.

It is difficult to accurately test if the implementation produces correct results
but the basic tests indicate that the implementation is most likely correct.
Probabilities add to one and results are rational in terms of what is expected.
Furthermore the formulas are relatively straightforward so it is easy to chech in
terms of coding the corectness. Finally we get the same results from each implementation
and taking that each implementation is independent of each other this is another
indication for a correct implementation.
'''

def load_corpus(input_path: str) -> str:
    """
    :param String input_path: gets the path of corpus file and loads it.
    :return String corpus: return the whole corpus
    """
    circlefile = open(input_path, encoding="utf8")
    corpus = circlefile.read()
    circlefile.close()

    # Print a slice of the corpus
    #print('--------CORPUS SAMPLE-----------')
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

    for a in range(1, 20, 1):
        alpha = a / 2000
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

def get_min_cross_entropy(res):
    minKey = min(res, key = res.get)
    min_ce = res[minKey][0]
    min_perplexity = res[minKey][1]
    return minKey, min_ce,min_perplexity

class KneserNeyDigrams:
    def __init__(self, words_seq, unigram_freqs, bigram_freqs, d):
        self._unigrams = unigram_freqs
        self._bigrams = bigram_freqs
        self._cache = {}
        
        self._after = nltk.defaultdict(float)
        self._before = nltk.defaultdict(float)
        self._col_sums = nltk.defaultdict(float)
        self._length = 0 #len(words_seq)
        self._d = d
            
        for word_prev, word in self._bigrams:
            self._after[word_prev] += 1 # equiv. to adjacancy matrix row sums
            self._length += 1 # equiv. to adjacancy matrix NNZ
            self._before[word] += 1 #equiv. to adjacancy matrix col sums.
            
    def compute_prob(self, bigram):
        
        if bigram in self._cache:
            return self._cache[bigram]
        else:
            w_prev, w = bigram
            if bigram in self._bigrams:
                prob = (self._bigrams[bigram] - self._d) / self._unigrams[w_prev]
            elif w_prev in self._unigrams and w in self._before:
                after = self._after[w_prev]
                before = self._before[w]
                
                alpha = (after * self._d) / self._unigrams[w_prev]
                # Here instead of after in the denominator we want col sums of non zero cols at row word_prev
                #Prev = before / (self._length - after )
                # Note: non zero cols indices = keys of self._before
                if w_prev not in self._col_sums:
                    n = 0
                    for w in self._before.keys():
                        if (w_prev, w) in self._bigrams:
                            n += self._before[w]
                    self._col_sums[w_prev] = n
                Prev = before / (self._length - self._col_sums[w_prev])
                prob = alpha * Prev
            else:
                prob = 0.0
                
            self._cache[bigram] = prob
            return prob
            
    def __str__(self):
        return 'D = {0:.3f}, Number of words = {1:d}'.format(self._d, self._length)

# Read Coprus
#input_path = r'C:\Users\User\Desktop\MSc Courses\Untitled Folder\europarl-v7.el-en.en'
input_path = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\Assignment 1\europarl-v7.el-en.en'

corpus1 = load_corpus(input_path)
corpus = corpus1#[0:1000000]

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
unigram_freq_train = get_unigram_freq(train_corpus)

# Get the mapping for the words we will keep (tokens with count >= threshold)
mapping = filter_by_freq_words(unigram_freq_train)

test_sent_tokenized_bigram = tokenize_sentences_and_padding(
    corpus_sentences=test_sent_tokenized,
    mapping=mapping,
    ngram='bigram', use_padding=True
    )

#test_sent_tokenized_trigram = tokenize_sentences_and_padding(
#    corpus_sentences=test_sent_tokenized,
#    mapping=mapping,
#    ngram='trigram', use_padding=True
#    )

# --------------
# BIGRAM MODEL:
# --------------

# Create unigram/bigram/trigram tokenization
train_sent_tokenized_bigram = tokenize_sentences_and_padding(
    corpus_sentences=train_sent_tokenized,
    mapping=mapping,
    ngram='bigram'
    )
dev_sent_tokenized_bigram = tokenize_sentences_and_padding(
    corpus_sentences=dev_sent_tokenized,
    mapping=mapping,
    ngram='bigram'
    )

# print('----SAMPLE OF TOKENIZED BIGRAMS----')
# print(train_sent_tokenized_bigram[0:10])

# Flatten the training tokinized corpus and create unigram and bigram frequencies dictionaries
sentences_tokenized_bigram_flattened = [val for sublist in train_sent_tokenized_bigram for val in sublist]

ce_perpl_results_bigram = get_cross_entropy_perplexity(
    model='bigram',
    words=sentences_tokenized_bigram_flattened,
    dev_sentences_set=dev_sent_tokenized_bigram
    )


plot_perplexity(ce_perpl_results_bigram)

alpha_bi, value_bi, value_perplexity = get_min_cross_entropy(ce_perpl_results_bigram)
print('-----BI-GRAMS MODEL-----')
print('Cross Entropy {}'.format(value_bi))
print('Perplexity {}'.format(value_perplexity))
print('a smoothing {}'.format(alpha_bi))

### TAKE 5: Adapting nltk code for trigrams.
words = sentences_tokenized_bigram_flattened
unigram_freq = nltk.FreqDist(words)
bigrams = nltk.bigrams(words)
bigram_freq = nltk.FreqDist(bigrams)
del bigram_freq[('end1', 'start1')]
vocab_size = len(unigram_freq)

d=0.7
time_start = time.perf_counter()
kn = KneserNeyDigrams(words, unigram_freq, bigram_freq, d)
print('')
print(kn)
#print('NNZ = {0:d}'.format(adj_csr.nnz))
print('')
for sentence in dev_sent_tokenized_bigram:
    for idx1,word1 in enumerate(sentence):
      if word1 == '1089':
          sentence[idx1] = 'UNK'

sequence_len = 0
entropy_kn = 0
for sentence in dev_sent_tokenized_bigram:
    for i in range(1, len(sentence)):
        previous_word = sentence[i - 1]
        word = sentence[i]
        
        entropy_kn -= np.log2(kn.compute_prob((previous_word, word)))
        sequence_len += 1
entropy_kn /= sequence_len
time_stop = time.perf_counter()
time_elapsed = time_stop - time_start
print('KN (6): Entropy = {0:.3f}, Perplexity = {1:.3f}, Time elapsed = {2:.3f} secs'.format(entropy_kn, 2 ** entropy_kn, time_elapsed))
print('-----BI-GRAMS MODEL-----')
print('Cross Entropy {}'.format(value_bi))
print('Perplexity {}'.format(value_perplexity))
print('a smoothing {}'.format(alpha_bi))





# 4b. Kneser Ney with sparse adjacancy matrix
##### TAKE FINAL #####
## This one is used for calculating
words = sentences_tokenized_bigram_flattened
unigram_freq = nltk.FreqDist(words)
bigrams = nltk.bigrams(words)
bigram_freq = nltk.FreqDist(bigrams)
del bigram_freq[('end1', 'start1')]
vocab_size = len(unigram_freq)

# Create a list with the distinct words of the (training) corpus    
distinct_words = list(set(words))
print('Distinct words = {0:d}'.format(len(distinct_words)))

num_distinct_words = len(distinct_words)
d = 0.5

time_start = time.perf_counter()
adj_rows = [] # Holds the row index of the non zero element of the adjacancy matrix.
adj_cols = [] # As above for columns.
adj_data = [] # The values of the non zero elements.
for i in range(num_distinct_words):
    word_prev = distinct_words[i]
    for j in range(num_distinct_words):
        word = distinct_words[j]
        if bigram_freq[(word_prev, word)] > 0:
            adj_rows.append(i)
            adj_cols.append(j)
            adj_data.append(1)
# Convert to NumPy arrays, to pass them into the sparse matrix constructor.
adj_rows = np.array(adj_rows)
adj_cols = np.array(adj_cols)
adj_data = np.array(adj_data)
# Create sparse matrix.
adj_coo = sparse.coo_matrix((adj_data, (adj_rows, adj_cols)), shape = (num_distinct_words, num_distinct_words), dtype = np.bool_)
adj_csc = adj_coo.tocsc() # This sparse matrix format is optimized for fast column operations.
adj_csr = adj_coo.tocsr() # As above, for row operations.

# We need as set of all column indices of the adjacancy matrix, to compute
# the denominators of Prev(w_{k}) later.
cols_set = set(range(num_distinct_words))
prev_k_denominators = np.zeros(num_distinct_words)
time_start1 = time.perf_counter()
for i in range(num_distinct_words):
    # We cannot use Boolean slicing with sparse matrices, as we do with NumPy matrices
    # thus in order to find the indices of the zero columns of the i-th row we do
    # the following:
    non_zero_cols = adj_csr.indices[adj_csr.indptr[i] : adj_csr.indptr[i + 1]] # This command returns a NumPy array with the NON ZERO col indices of row i.
    non_zero_cols_set = set(non_zero_cols) # Convert the NumPy array to Python core set.
    zero_cols_set = cols_set - non_zero_cols_set # Taking the set difference with the set of all column indices gives us what we want.
    zero_cols_indices = np.array(list(zero_cols_set)) # Finnaly we convert the set back to a NumPy array.
    # Now we have the desired indices we can compute the i-th denominator, in a similar manner to the NumPy implementation.
    prev_k_denominators[i] = np.sum(adj_csc[:, zero_cols_indices]) 
    if (i+1) % 1000 == 0:
        tmp = time.perf_counter() - time_start1
        print('Loop: {0:.0f}/{1:.0f}, Time Passed:{2:.3f}, Estimated Time:{3:.3f}'.format(i+1,num_distinct_words,tmp,(tmp/(i+1))*num_distinct_words))



#--- Check for common words
dev_sentences_tokenized_bigram_flattened = [val for sublist in dev_sent_tokenized_bigram for val in sublist]
z=nltk.FreqDist(dev_sentences_tokenized_bigram_flattened)
z2 = nltk.FreqDist(words)
ncwords = [w for w in z if w not in z2]


#--- Code for changing 1089---
for sentence in dev_sent_tokenized_bigram:
    for idx1,word1 in enumerate(sentence):
      if word1 == '1089':
          sentence[idx1] = 'UNK'



sequence_len = 0
entropy_kn = 0
cnt = 0
for sentence in dev_sent_tokenized_bigram:
    time_start1 = time.perf_counter()
    for i in range(1, len(sentence)):
        previous_word = sentence[i - 1]
        word = sentence[i]    
        k_prev = distinct_words.index(previous_word)
        k = distinct_words.index(word)
        # Here we obtain a NumPy array (cols) with the indices of the non zero elements in row k_prev,
        # i.e. we have adjacancy_matrix[k_prev, k] = 1
        cols = adj_csr.indices[adj_csr.indptr[k_prev] : adj_csr.indptr[k_prev + 1]]
        # Therefore, the condition of whether there exists a "connection" of row k_prev and col k, becomes:
        if k in cols:
            digram_prob = (bigram_freq[(previous_word, word)] - d) / unigram_freq[previous_word]
        else:
            # Same steps with NumPy implementation.
            prev_k = np.sum(adj_csc[:, k]) # Summing over k column, use the optimized column operation format.
            alpha = d * np.sum(adj_csr[k_prev, :]) / unigram_freq[previous_word] # Summing over k_prev row, use optimized row operation format.
            digram_prob = alpha * prev_k / prev_k_denominators[k_prev]
        
        entropy_kn -= np.log2(digram_prob)
        sequence_len += 1

    cnt =cnt + 1
    # Below code is for timing purposes to evaluate the time needed
    if (cnt) % 500 == 0:
        tmp = time.perf_counter() - time_start1
        print('Loop: {0:.0f}/{1:.0f}, Time Passed:{2:.3f}, Estimated Time:{3:.3f}'.format(cnt,len(dev_sent_tokenized_bigram),tmp,(tmp/cnt)*len(dev_sent_tokenized_bigram)))

entropy_kn /= sequence_len

time_stop = time.perf_counter()
time_elapsed = time_stop - time_start
print('KN (1): Entropy = {0:.3f}, Perplexity = {1:.3f}, Time elapsed = {2:.3f} secs'.format(entropy_kn, 2 ** entropy_kn, time_elapsed))
print('-----BI-GRAMS MODEL-----')
print('Cross Entropy {}'.format(value_bi))
print('Perplexity {}'.format(value_perplexity))
print('a smoothing {}'.format(alpha_bi))



# --------------------------------------------------------------
# Kneser Ney smoothing Revisited Version on Numpy implementation
# Ver 2.0 Numpy Implementation
# --------------------------------------------------------------

words = sentences_tokenized_bigram_flattened
unigram_freq = nltk.FreqDist(words)
bigrams = nltk.bigrams(words)
bigram_freq = nltk.FreqDist(bigrams)

# Create a list with the distinct words of the (training) corpus    
distinct_words = list(set(words))
print('Distinct words = {0:d}'.format(len(distinct_words)))

# 4. KN smoothing revisited.
# Step 1: Create adjacancy matrix of word bigrams
num_distinct_words = len(distinct_words)
d = 0.75
time_start = time.perf_counter()
adjacancy_matrix = np.zeros((num_distinct_words, num_distinct_words), dtype = np.bool_) # bool_ is represented by one byte.
for i in range(num_distinct_words):
    word_prev = distinct_words[i]
    for j in range(num_distinct_words):
        word = distinct_words[j]
        if bigram_freq[(word_prev, word)] > 0:
            adjacancy_matrix[i][j] = 1
            
# Step 2: use adjacancy matrix to compute the denumerators of the expression
#         of Prev(w_{k}) in the slides.
time_start1 = time.perf_counter()
prev_k_denumerators = np.zeros(num_distinct_words)
for i in range(num_distinct_words):
    zero_cols = adjacancy_matrix[i] == 0
    prev_k_denumerators[i] = np.sum(adjacancy_matrix[:, zero_cols] == 1)
    if (i+1) % 100 == 0:
        tmp = time.perf_counter() - time_start1
        print('Loop: {0:.0f}/{1:.0f}, Time Passed:{2:.3f}, Estimated Time:{3:.3f}'.format(i,num_distinct_words,tmp,(tmp/(i+1))*num_distinct_words))
    
# Step 3: Compute entropy and perplexity of the dev set.
sequence_len = 0
entropy_kn = 0
cnt = 0
time_start1 = time.perf_counter()
for sentence in dev_sent_tokenized_bigram:    
    for i in range(1, len(sentence)):
        previous_word = sentence[i - 1]
        word = sentence[i]
        
        k_prev = distinct_words.index(previous_word)
        k = distinct_words.index(word)
        
        if adjacancy_matrix[k_prev, k]:
            digram_prob = (bigram_freq[(previous_word, word)] - d) / unigram_freq[previous_word]
        else:
            prev_k = np.sum(adjacancy_matrix[:, k] == 1)
            alpha = d * np.sum(adjacancy_matrix[k_prev, :] == 1) / unigram_freq[previous_word]
            digram_prob = alpha * prev_k / prev_k_denumerators[k_prev]
            
        entropy_kn -= np.log2(digram_prob)
        sequence_len += 1
    cnt =cnt + 1
    if (cnt) % 500 == 0:
        tmp = time.perf_counter() - time_start1
        print('Loop: {0:.0f}/{1:.0f}, Time Passed:{2:.3f}, Estimated Time:{3:.3f}'.format(cnt,len(dev_sent_tokenized_bigram),tmp,(tmp/cnt)*num_distinct_words))

    
entropy_kn /= sequence_len
time_stop = time.perf_counter()
time_elapsed = time_stop - time_start
print('\n Kneser-Ney: Entropy = {0:.3f}, Peprlexity = {1:.3f}, Time elapsed = {2:.3f}'.format(entropy_kn, 2 ** entropy_kn, time_elapsed))
print('-----BI-GRAMS MODEL-----')
print('Cross Entropy {}'.format(value_bi))
print('Perplexity {}'.format(value_perplexity))
print('a smoothing {}'.format(alpha_bi))

# ------------------------------------------------------
# Kneser Ney smoothing Original Version based on theory
# Very slow
# Version 1.0
# ------------------------------------------------------

unigram_freq = nltk.FreqDist(sentences_tokenized_bigram_flattened)
bigrams = nltk.bigrams(sentences_tokenized_bigram_flattened)
bigram_freq_train = nltk.FreqDist(bigrams)
dev_sentences_tokenized_bigram_flattened = [val for sublist in dev_sent_tokenized_bigram for val in sublist]
bigram_freq = nltk.FreqDist(nltk.bigrams(dev_sentences_tokenized_bigram_flattened))
del bigram_freq[('end1', 'start1')]

vocab_size = len(unigram_freq)
d = 0.75
sum_prob1 = 0
unseen_bigrams = {}
total_count = 0

# Robin hood action! Steal mass probability from the rich ones!
# Find all development set bigrams that exist in the corpus and calculate sum (log(probability))
for sent in dev_sent_tokenized_bigram:
    for idx in range(1,len(sent)):
       if bigram_freq_train[(sent[idx-1], sent[idx])]>0: #For existing bigrams in test set
           prob1 = (bigram_freq_train[(sent[idx-1], sent[idx])]-d) / (unigram_freq[(sent[idx-1])])
           sum_prob1 += math.log2(prob1)
           total_count += 1
       else: # Put the not existing ones to calculate below their contribution
         #print(bigram_freq[(sent[idx-1], sent[idx])])
         unseen_bigrams.update({(sent[idx-1], sent[idx]): 0})

# For all not seen bigrams calculate probability according to advance Knesser Ney approach
# by returning stolen probability mass
print('Starting step 2...')
sum_prob2 = 0
time_start = time.perf_counter()
i = 0
tmp1 = len(unseen_bigrams)
for twogram in unseen_bigrams:
    # calculate a(w_(k-1))
    aprop1 = (d/unigram_freq[twogram[0]])
    # Find all words in the vocabulary that exist in a diagram with w_(k-1)
    lst_tmp = {w for w in unigram_freq if bigram_freq_train[(twogram[0],w)]>0}
    # Calculate proportion
    aprop = aprop1*len(lst_tmp)
    # FInd prev(w_k) nominator from Prev(w_k)
    lst_tmp = {w for w in unigram_freq if bigram_freq_train[(w,twogram[1])]>0} # prev(w_k)
    # Find all bigrams that do not have the word w_k in the corpus
    lst_tmp1 = {w for w in unigram_freq if bigram_freq_train[(twogram[0],w)]==0} # All bigram that do not have W_(k-1)
    total = 0 # total for sum zeroed at each loop for each word
    i += 1 # for monitoring purposes
    # For all these words calculate the sum of all sets of words that do not have a bigram with word w_(k-1)
    for word in lst_tmp1:
      lst_tmp2 = {w for w in unigram_freq if bigram_freq_train[(w,word)]>0}
      total = total + len (lst_tmp2)
    # Now that we have everything calculate probabilities
    prob2= aprop * len(lst_tmp)/total
    sum_prob2 += math.log2(prob2)
    total_count += 1
    # Just to monitor some timing
    if i % 100 == 0:
        tm1 = (time.perf_counter() - time_start)/i
        print('Loop number:%.0f/%.0f, estimation:%2.f, left:%2f' % (i,tmp1,tm1*tmp1,(tmp1-i)*tm1))
    
time_stop = time.perf_counter()
time_elapsed1 = time_stop - time_start    
print('Total time:%.3f' % time_elapsed1)    

HC = -(sum_prob1+sum_prob2) / total_count
perpl = math.pow(2, HC)
print('Cross Entropy {}'.format(HC))
print('Perplexity {}'.format(perpl))
#Laplace smoothing optimal
print('Cross Entropy {}'.format(value_bi))
print('Perplexity {}'.format(value_perplexity))


# ----------------
# TRIGRAM MODEL:
# ----------------

# Create unigram/bigram/trigram tokenization
# Get train and test sets:
train_sent_tokenized_trigram = tokenize_sentences_and_padding(
    corpus_sentences=train_sent_tokenized,
    mapping=mapping,
    ngram='trigram'
    )

dev_sent_tokenized_trigram = tokenize_sentences_and_padding(
    corpus_sentences=dev_sent_tokenized,
    mapping=mapping,
    ngram='trigram'
    )

# print('The training corpus contains {} sentences'.format(len(train_sent_tokenized_trigram)))
# print('The developement corpus contains {} sentences'.format(len(dev_sent_tokenized_trigram)))

#print('----SAMPLE OF TOKENIZED TRIGRAMS----')
#print(train_sent_tokenized_trigram[0:10])

# Flatten the training tokinized corpus and create unigram and bigram frequencies dictionaries
sentences_tokenized_trigram_flattened = [val for sublist in train_sent_tokenized_trigram for val in sublist]

ce_perpl_results_tri = get_cross_entropy_perplexity(
    model='trigram',
    words=sentences_tokenized_trigram_flattened,
    dev_sentences_set=dev_sent_tokenized_trigram
)

plot_perplexity(ce_perpl_results_tri)

alpha_tri, value_tri, value_perplexity_tri = get_min_cross_entropy(ce_perpl_results_tri)
print('-----TRI-GRAMS MODEL-----')
print('Cross Entropy {}'.format(value_tri))
print('Perplexity {}'.format(value_perplexity_tri))
print('a smoothing {}'.format(alpha_tri))

trigrams = nltk.trigrams(sentences_tokenized_trigram_flattened)
trigram_freq = nltk.FreqDist(trigrams)
remove = [k for k in trigram_freq.keys() if k[2] in ['start1', 'start2']]
for k in remove: del trigram_freq[k]    
dev_sentences_tokenized_trigram_flattened = [val for sublist in dev_sent_tokenized_trigram for val in sublist]
trigrams_dev = list(nltk.trigrams(dev_sentences_tokenized_trigram_flattened))
trigram_dev_freq = nltk.FreqDist(trigrams_dev)
remove = [k for k in trigram_dev_freq.keys() if k[2] in ['start1', 'start2']]
for k in remove: del trigram_dev_freq[k]    
len(trigrams_dev)
len(dev_sentences_tokenized_trigram_flattened)
sum_prob = 0
trigram_cnt = 0
g=0
kn_tri = nltk.KneserNeyProbDist(trigram_freq)
kn_tri.samples()
kn_tri.max()
for itm in trigrams_dev:
        if kn_tri.prob(itm) != 0:
          sum_prob += math.log2(kn_tri.prob(itm))
        else:
          g = g+1            
        trigram_cnt += 1
HC = -sum_prob / trigram_cnt
perpl = math.pow(2, HC)

print("Cross Entropy: {0:.3f}".format(HC))
print("perplexity: {0:.3f}".format(perpl))
print("g: {0:.3f}".format(g))
g/len(trigrams_dev)