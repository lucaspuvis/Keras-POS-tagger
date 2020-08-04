import os
import keras
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

from conllu import parse_incr
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from tensorflow.python.client import device_lib
from keras.optimizers import Adam
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

print(device_lib.list_local_devices())
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

def import_data(filepath):
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



def generate_unique_indices(sentences, sentence_tags):
    # Acquiring all unique words and tags
    words = set([w.lower() for s in sentences for w in s])
    tags = set([t for ts in sentence_tags for t in ts])

    # Converting words and tags to unique indexes
    word2index = {w: i + 2 for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0     # Used for padding
    word2index['-OOV-'] = 1     # Value for OOVs

    tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
    tag2index['-PAD-'] = 0

    return word2index, tag2index



def convert_to_integers(sentences, sentence_tags, word2index, tag2index):
    sentences_X, tags_y = [], []

    for s in sentences:
        s_int = []
        for w in s:
            try: 
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])

        sentences_X.append(s_int)

    for s in sentence_tags:
        tags_y.append([tag2index[t] for t in s])


    return sentences_X, tags_y



def pad_sentences_tags(sentences_X, tags_y):
    MAX_LENGTH = len(max(sentences_X, key=len))

    padded_sentences_X = pad_sequences(sentences_X, maxlen=MAX_LENGTH, padding='post')
    padded_tags_y = pad_sequences(tags_y, maxlen=MAX_LENGTH, padding='post')

    return padded_sentences_X, padded_tags_y, MAX_LENGTH



def convert_to_words(sentences_X, tags_y, word2index, tag2index):
    sentences, sentence_tags = [], []

    index2word = {v: k for k, v in word2index.items()}
    index2tag = {v: k for k, v in tag2index.items()}

    for s in sentences_X:
        sentences.append([index2word[t] for t in s])


    for s in tags_y:
        tags_y.append([index2tag[t] for t in s])
        
    return sentences, sentence_tags



# Next two functions are blatantly stolen from
# https://nlpforhackers.io/lstm-pos-tagger-keras/
def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)



def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences




def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy




def train_model(train_sentences_X, train_tags_y, word2index, tag2index, batch_size, epochs, MAX_LENGTH):
    model = Sequential()
    model.add(Embedding(len(word2index), 128))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(len(tag2index))))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy', ignore_class_accuracy(0)])
    model.summary()

    model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=batch_size, epochs=epochs, validation_split=0.2)

    return model



def evaluate_model(model, test_sentences_X, test_tags_y, tag2index):
    scores = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
    print(f"{model.metrics_names[2]}: {scores[2] * 100}")
    return scores[2] * 100
    tag2index_copy = tag2index.copy()
    tag2index_copy.pop("-PAD-")
    index2tag = {i: t for t, i in tag2index.items()}

    y_pred = model.predict(test_sentences_X)
    y_pred = logits_to_tokens(y_pred, index2tag)
    
    conv_test_tags_y = []
    for test_tags in test_tags_y:
        conv_test_tags_y.append([index2tag[t] for t in test_tags])

    unpadded_y_pred, unpadded_test_tags_y = [], []
    for pred_tags, test_tags in zip(y_pred, conv_test_tags_y):
        unpad_pred, unpad_test = [], []
        for pred_tag in pred_tags:
            if pred_tag != "-PAD-":
                unpad_pred.append(pred_tag)

        for test_tag in test_tags:
            if test_tag != "-PAD-":
                unpad_test.append(test_tag)
        
        unpadded_y_pred.append(unpad_pred)
        unpadded_test_tags_y.append(unpad_test)
        

    heatmap = {i: {i: 0 for i in tag2index_copy} for i in tag2index_copy}

    total_tag_count, total_correct_tag_count = 0, 0
    for pred_tags, test_tags in zip(unpadded_y_pred, unpadded_test_tags_y):
        for pred_tag, test_tag in zip(pred_tags, test_tags):
            heatmap[pred_tag][test_tag] += 1
            total_tag_count += 1
            if pred_tag == test_tag:
                total_correct_tag_count += 1
    
    sentence_accuracy = total_correct_tag_count / total_tag_count
    
    df = pd.DataFrame(heatmap)
    # df.iloc[:,:] = Normalizer(norm='l1').fit_transform(df)
    print("Token accuracy: {}".format(sentence_accuracy))

    sn.set()
    sn.heatmap(df, annot=True, fmt="d", linewidths=.5)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()




if __name__ == '__main__':
    LOAD_MODEL = False
    epoch = 10
    batch_size = 32

    if LOAD_MODEL:
        # There's no feature checking whether a model file exists, so this can result in an error
        model = keras.models.load_model("lstm.h5")

        data = import_data("fr_gsd-ud-train.conllu")
        indices = generate_unique_indices(*data) # Training data and tags
        converted_data = convert_to_integers(*data, *indices)
        padded_sentences_X, padded_tags_y, MAX_LENGTH = pad_sentences_tags(*converted_data)

        word2index, tag2index = indices
        train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = train_test_split(padded_sentences_X, padded_tags_y, test_size=0.2)

        evaluate_model(model, test_sentences_X, test_tags_y, tag2index)
    else:
        data = import_data("fr_gsd-ud-train.conllu")
        indices = generate_unique_indices(*data) # Training data and tags
        converted_data = convert_to_integers(*data, *indices)
        padded_sentences_X, padded_tags_y, MAX_LENGTH = pad_sentences_tags(*converted_data)
        word2index, tag2index = indices
        train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = train_test_split(padded_sentences_X, padded_tags_y, test_size=0.2)

        model = train_model(train_sentences_X, train_tags_y, word2index, tag2index, batch_size, epoch, MAX_LENGTH)
        score = evaluate_model(model, test_sentences_X, test_tags_y, tag2index)
        model.save("lstm.h5")

