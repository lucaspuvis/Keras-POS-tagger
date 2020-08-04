# --------------------- FOREWORD --------------------- #
# Code for importing data is a modified version of     #
# the tutorial originally supplied in the handout      #
# https://nlpforhackers.io/lstm-pos-tagger-keras/      #
# Code for training and testing the model is written   #
# by myself, Lucas Puvis de Chavannes.                 #
# ---------------------------------------------------- #

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import math
from conllu import parse_incr
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import Normalizer

class hmm_model():
    def __init__(self, word2index, tag2index):
        self.word2index = word2index
        self.tag2index = tag2index
        self.accuracy = 0.0
        self.tag_occurence_dict = dict()
        self.tag_sequence_dict = dict(dict())
        self.tag_word_occurence_dict = dict(dict())


    def train(self, train_sentences_X, train_tags_y, smooth=None):
        '''
        Trains the HMM model with options for smoothing.

        Parameters:
        train_sentences_X (list<int>): A list of sentences converted into an integer representation
        train_tags_y (list<int>): The list of corresponding sentence tags converted into an integer representation
        smooth (str): Smoothing technique. Options are addone. Defaults to None.
        '''
        self.tag_sequence_dict = {self.tag2index[i]: dict() for i in self.tag2index}
        self.tag_word_occurence_dict = {self.tag2index[i]: dict() for i in self.tag2index}

        if smooth == "addone":
            self.tag_occurence_dict = {self.tag2index[i]: 1 for i in self.tag2index}
        else:
            self.tag_occurence_dict = {self.tag2index[i]: 0 for i in self.tag2index}

        # Dict comprehenception? I am big brain man :) :big_thonkin:
        # self.tag_sequence_dict = {self.tag2index[i]: {self.tag2index[i]: 0 for i in self.tag2index} for i in self.tag2index}
        # self.tag_word_occurence_dict = {self.tag2index[i]: {self.word2index[i]: 0 for i in word2index} for i in self.tag2index}

        for key in self.tag_sequence_dict:
            if smooth == "addone":
                self.tag_sequence_dict[key] = {self.tag2index[i]: 1 for i in self.tag2index}
            else:
                self.tag_sequence_dict[key] = {self.tag2index[i]: 0 for i in self.tag2index}

        for key in self.tag_word_occurence_dict:
            if smooth == "addone":
                self.tag_word_occurence_dict[key] = {self.word2index[i]: 1 for i in self.word2index}
            else:
                self.tag_word_occurence_dict[key] = {self.word2index[i]: 0 for i in self.word2index}

        prev_tag = 0
        for i, (sentence, sentence_tags) in enumerate(zip(train_sentences_X, train_tags_y)):
            for token, tag in zip(sentence, sentence_tags):
                self.tag_occurence_dict[tag] += 1
                self.tag_word_occurence_dict[tag][token] += 1

                # For tag sequences we need to start at the second tag
                if i > 0:
                    self.tag_sequence_dict[prev_tag][tag] += 1

                prev_tag = tag


    
    def predict(self, sentences_X):

        for sentence in sentences_X:
            tag_sequence = []

            prev_tag = self.tag2index["-START-"]
            for i, token in enumerate(sentence):
                # Skip START and END
                if i is 0 or i is len(sentence)-1:
                    continue

                tag_probs = dict()
                
                for tag in self.tag2index:
                    # We skip the first three tags PAD, START and END
                    if tag == "-START-" or tag == "-END-" or tag == "-PAD-":
                        continue

                    tag_seq = self.tag_sequence_dict[prev_tag][tag2index[tag]]
                    tag_pre = self.tag_occurence_dict[prev_tag]

                    wrd_tag = self.tag_word_occurence_dict[tag2index[tag]][token]
                    tag_occ = self.tag_occurence_dict[tag2index[tag]]
                    
                    if tag_seq == 0 or tag_pre == 0 or wrd_tag == 0 or tag_occ == 0:
                        print(tag_seq)
                        print(tag_pre)
                        print(wrd_tag)
                        print(tag_occ)

                    tag_probs[tag2index[tag]] = math.exp((math.log(wrd_tag) / math.log(tag_occ)) * (math.log(tag_seq) / math.log(tag_pre)))
                
                tag_sequence.append(max(tag_probs, key=tag_probs.get))

            # START and END are guaranteed, so append them
            tag_sequence.insert(0, tag2index["-START-"])
            tag_sequence.append(tag2index["-END-"])
            yield tag_sequence


    def measure_overall_accuracy(self, test_sentences_X, test_tags_y, verbose=False):
        tags_y = self.predict(test_sentences_X)
        correct_estimates = 0

        for tags, test_tags in zip(tags_y, test_tags_y):
            if tags == test_tags:
                correct_estimates += 1

        self.accuracy = correct_estimates/len(test_tags_y)
        return self.accuracy


    def measure_token_accuracy(self, test_sentences_X, test_tags_y):
        tags_y = self.predict(test_sentences_X)
        predicted_tags, test_tags = [], []

        index2tag = {v: k for k, v in tag2index.items()}

        for tag_seq_pred, tag_seq_test in zip(tags_y, test_tags_y):
            predicted_tags.append([index2tag[t] for t in tag_seq_pred])
            test_tags.append([index2tag[t] for t in tag_seq_test])


        # Clean up unwanted tags
        for tag_seq_pred, tag_seq_test in zip(predicted_tags, test_tags):

            tag_seq_pred.remove("-START-")
            tag_seq_pred.remove("-END-")

            tag_seq_test.remove("-START-")
            tag_seq_test.remove("-END-")

        tag2index_copy = self.tag2index.copy()
        tag2index_copy.pop("-PAD-")
        tag2index_copy.pop("-START-")
        tag2index_copy.pop("-END-")

        heatmap = {i: {i: 0 for i in self.tag2index} for i in self.tag2index}

        total_tag_count, total_correct_tag_count = 0, 0
        for tags, test_tags in zip(predicted_tags, test_tags):
            for tag, test_tag in zip(tags, test_tags):
                heatmap[tag][test_tag] += 1
                total_tag_count += 1
                if tag == test_tag:
                    total_correct_tag_count += 1
        
        sentence_accuracy = total_correct_tag_count / total_tag_count
        
        df = pd.DataFrame(heatmap)
        df.iloc[:,:] = Normalizer(norm='l1').fit_transform(df)
        print("Token accuracy: {}".format(sentence_accuracy))

        sns.set()
        sns.heatmap(df, annot=True, fmt="d", linewidths=.5)

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    

def import_data(filepath):
    '''
    Imports the data from the specific .conllu file supplied.

    Parameters:
    filepath (str): Filepath to conllu file

    Returns:
    sentences (list<str>): A list of sentences
    sentence_tags (list<str>): A list of tags

    '''
    data_file = open(filepath, mode="r", encoding="utf8")
    tokenlist = list(parse_incr(data_file))

    tagged_sentences = []
    for sentence in tokenlist:
        tmp = []
        for token in sentence:
            tmp.append((token["form"], token["upos"]))
        
        tagged_sentences.append(tmp)

    sentences, sentence_tags = [], []

    for tagged_sentence in tagged_sentences:
        sentence, tags = zip(*tagged_sentence)
        sentences.append(np.array(sentence))
        sentence_tags.append(np.array(tags))

    return sentences, sentence_tags



def treebank_data():
    '''
    Uses the nltk treebank data to test with

    Returns:
    sentences (list<str>): A list of sentences
    sentence_tags (list<str>): A list of tags

    '''
    nltk.download('treebank')

    tagged_sentences = nltk.corpus.treebank.tagged_sents()
    sentences, sentence_tags = [], []

    for tagged_sentence in tagged_sentences:
        sentence, tags = zip(*tagged_sentence)
        sentences.append(np.array(sentence))
        sentence_tags.append(np.array(tags))

    return sentences, sentence_tags



def generate_unique_indices(sentences, sentence_tags):
    '''
    Assigns all unique words a unique integer

    Parameters:
    sentences (list<str>): The training data sentences
    sentence_tags (list<str>): The training data tags

    Returns:
    word2index (dict): A dictionary mapping each unique word to a unique integer
    tag2index (dict): A dictionary mapping each unique tag to a unique integer
    '''

    # Acquiring all unique words and tags
    words = set([w.lower() for s in sentences for w in s])
    tags = set([t for ts in sentence_tags for t in ts])

    # Converting words and tags to unique indexes
    word2index = {w: i + 4 for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0     # Used for padding
    word2index['-START-'] = 1   # Denotes start of sentence
    word2index['-END-'] = 2     # Denotes end of sentence
    word2index['-OOV-'] = 3     # Value for OOVs

    tag2index = {t: i + 3 for i, t in enumerate(list(tags))}
    tag2index['-PAD-'] = 0
    tag2index['-START-'] = 1
    tag2index['-END-'] = 2

    return word2index, tag2index



def convert_to_integers(sentences, sentence_tags, word2index, tag2index):
    '''
    Generates a list of sentences and tags based on the unique indices

    Parameters:
    sentences (list<str>): A list of sentences
    sentence_tags (list<str>): A list of tags corresponding to the sentences
    word2index (dict<str:int>): A dict of unique word to integer mapping
    tag2index (dict<str:int>): A dict of unique tag to integer mapping
    test_size (int): Size of the test sample after split. Default value is 0.2 (20%)

    Returns:
    sentences_X (list<int>): The converted list of sentences
    tags_y (list<int>): The converted list of tags corresponding to the sentences
    '''
    sentences_X, tags_y = [], []

    for s in sentences:
        s_int = []
        s_int.append(word2index['-START-'])
        for w in s:
            try: 
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])

        s_int.append(word2index['-END-'])
        sentences_X.append(s_int)


    for s in sentence_tags:
        tags_y.append([tag2index[t] for t in s])

    for s in tags_y:
        s.insert(0, tag2index['-START-'])
        s.append(tag2index['-END-'])
        
    return sentences_X, tags_y



def pad_sentences_tags(sentences_X, tags_y):
    '''
    Pads all sentences (in integer form) with an appropriate indice to match the longest sentence.

    Parameters:
    sentences_X (list<int>): The list of sentences as ints
    tags_y (list<int>): The list of tag ints corresponding to the sentences

    Returns:
    padded_sentences_X (list<int>): The list of sentences as ints with padding
    padded_tags_y (list<int>): The list of tag ints corresponding to the sentences with padding

    '''
    MAX_LENGTH = len(max(sentences_X, key=len))

    padded_sentences_X = pad_sequences(sentences_X, maxlen=MAX_LENGTH, padding='post')
    padded_tags_y = pad_sequences(tags_y, maxlen=MAX_LENGTH, padding='post')

    return padded_sentences_X, padded_tags_y



def convert_to_words(sentences_X, tags_y, word2index, tag2index):
    sentences, sentence_tags = [], []

    index2word = {v: k for k, v in word2index.items()}
    index2tag = {v: k for k, v in tag2index.items()}

    for s in sentences_X:
        sentences.append([index2word[t] for t in s])


    for s in tags_y:
        tags_y.append([index2tag[t] for t in s])
        
    return sentences, sentence_tags



def get_data(conllu_path="", test_size=0.2):
    '''
    Helper function that grabs a specified corpus, generates unique indices for each token and tag,
    converts each word sentence and corresponding tag sentence from words/tags to integers from these indices.
    Lastly, it pads the data to make sure every sentence and tag sentence is of equal length, then splits it
    into training and test data.

    Parameters:
    conllu_path (str): Path to a conllu corpus file. Default is empty string
    test_size (real): Value between 0 and 1 determining percentage size of test sample

    Returns:
    train_sentences_X (list<int>): List of sentences in integers denoting their word, meant for training
    test_sentences_X (list<int>): List of sentences in integers denoting their word, meant for testing
    train_tags_y (list<int>): List of tags in integers denoting their tag corresponding to the sentences, meant for training
    test_tags_y (list<int>): List of tags in integers denoting their tag corresponding to the sentences, meant for testing
    word2index (dict<str:int>): A table of all unique words and their associated unique integer
    tag2index (dict<str:int>): A table of all unique tags and their associated unique integer
    tags (list<str>): A list of all unique tags
    '''
    if not conllu_path:
        data = treebank_data()
    else:
        data = import_data(conllu_path)
    indices = generate_unique_indices(*data) # Training data and tags
    converted_data = convert_to_integers(*data, *indices)
    # padded_data = pad_sentences_tags(*converted_data)

    word2index, tag2index = indices
    train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = train_test_split(*converted_data, test_size=test_size)

    return train_sentences_X, test_sentences_X, train_tags_y, test_tags_y, word2index, tag2index
 
    

if __name__ == "__main__":
    train_sentences_X, test_sentences_X, train_tags_y, test_tags_y, word2index, tag2index = get_data("fr_gsd-ud-train.conllu", test_size=0.2)
    hmm = hmm_model(word2index, tag2index)
    hmm.train(train_sentences_X, train_tags_y, smooth="addone")


    hmm.measure_token_accuracy(test_sentences_X, test_tags_y)
    print("Overall accuracy: {}".format(hmm.measure_overall_accuracy(test_sentences_X, test_tags_y)))
