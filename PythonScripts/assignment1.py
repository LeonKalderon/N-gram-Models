#Import essential libraries
import nltk
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from nltk import word_tokenize, sent_tokenize
import text_processing as tp

def load_corpus(input_path):
    #input_path = r'C:\Users\Georgia.Sarri\Documents\Msc\5th\TextAnalytics\Assignmnets\1st\europarl-v7.el-en.en'
    circlefile = open(input_path, encoding="utf8")
    corpus = circlefile.read()
    circlefile.close()
    # Print a slice of the corpus
    print(corpus[0:1000])

    sentences_corpus = sent_tokenize(corpus)

    train_corpus_sentences, test_and_developement_corpus_sentences = train_test_split(sentences_corpus, test_size=0.3,
                                                                                      random_state=1)
    developement_corpus_sentences, test_corpus_sentences = train_test_split(test_and_developement_corpus_sentences,
                                                                            test_size=0.5, random_state=1)

    print('The training corpus contains {} sentences'.format(len(train_corpus_sentences)))
    print('The developement corpus contains {} sentences'.format(len(developement_corpus_sentences)))
    print('The test corpus contains {} sentences'.format(len(test_corpus_sentences)))

    return train_corpus_sentences, test_and_developement_corpus_sentences, developement_corpus_sentences, test_corpus_sentences


def set_start_end_tokens(sentences, start1='start1', start2='start2', end='end1'):
    for i, sent in enumerate(sentences):
        sentences[i] = start1 + start2 + sent + end

    return sentences

def tokenize(train_corpus_sentences, developement_corpus_sentences):
    train_corpus = ' '.join(train_corpus_sentences)
    developement_corpus = ' '.join(developement_corpus_sentences)

    print(train_corpus[0:1000])

    train_tokens = nltk.word_tokenize(train_corpus)
    dev_tokens = nltk.word_tokenize(developement_corpus)

    return train_tokens, dev_tokens


def del_small_freq( freq_dict, freq=10, repl='UNK'):
   UNK_counter = 0
   UNK_keys = []
   if (freq_dict[repl] == 0):
        for token, count in list(freq_dict.items()):
            if count < freq:
                del freq_dict[token]
                UNK_keys.append(token)
                UNK_counter += 1
   return UNK_keys, UNK_counter

def replace(unigram_freq, train_token):
    print('Train Token size: ', len(train_tokens))
    replaced_tokens = train_token
    for i in range(len(train_tokens)):
        if unigram_freq[train_tokens[i]]<10:
            replaced_tokens[i] = 'UNK'

    replaced_freq = dict(nltk.FreqDist(replaced_tokens))

    return replaced_tokens, replaced_freq

def replace_2(unigram_freq):
    notunk = {k: v for k, v in unigram_freq.items() if v >= 10}

    unk1 = list(notunk)
    len(unk1)
    mapping = nltk.defaultdict(lambda: 'UNK')
    for v in unk1:
        mapping[v] = v

    replaced_tokens = [mapping[v] for v in train_tokens]
    replaced_freq = dict(nltk.FreqDist(replaced_tokens))

    return replaced_tokens, replaced_freq

def calculate_Laplace(word, freq_dict, corpus_length, V, alpha):
    return float(freq_dict[word]+alpha) / float(corpus_lenght + alpha*V)

if __name__== '__main__':
    input_path = r'C:\Users\Georgia.Sarri\Documents\Msc\5th\TextAnalytics\Assignmnets\1st\europarl-v7.el-en.en'
    train_corpus_sentences, test_and_developement_corpus_sentences, developement_corpus_sentences, test_corpus_sentences = load_corpus(input_path)
    START1_TOKEN = 'start1 '
    START2_TOKEN = 'start2 '
    END_TOKEN = ' end1'

    train_corpus_sentences = set_start_end_tokens(train_corpus_sentences)
    developement_corpus_sentences = set_start_end_tokens(developement_corpus_sentences)

    print(train_corpus_sentences[0:10])

    test_corpus = ' '.join(test_corpus_sentences)
    test_corpus = START1_TOKEN + START2_TOKEN + test_corpus + END_TOKEN

    train_tokens, dev_tokens = tokenize(train_corpus_sentences[0:10000], developement_corpus_sentences[0:10000])

    print(train_tokens[0:10])

    train_tokens = tp.normalize(train_tokens)
    dev_tokens = tp.normalize(dev_tokens)

    unigram_freq = dict(nltk.FreqDist(train_tokens))

    #replaced_tokens, replaced_freq = replace(unigram_freq, train_tokens)
    replaced_tokens, replaced_freq = replace_2(unigram_freq)
    print(replaced_tokens, replaced_freq)

'''
    corpus_lenght = 0;
    V = 0;
    for key in replaced_freq.keys():
        corpus_lenght+=replaced_freq[key]
        V+=1
    print(corpus_lenght, V)

    laplace_prob = {}
    for key in replaced_freq.keys():
        laplace_prob[key] = calculate_Laplace(key, replaced_freq,corpus_lenght, V, 0.1)

    print(laplace_prob)
'''
'''
    UNK_keys, UNK_counter = del_small_freq(unigram_freq)
    unigram_freq['UNK'] = UNK_counter
'''

