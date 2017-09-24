# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints

import tensorflow as tf

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


data_train = pd.read_csv(
    '/home/shiva/work/textClassifier/labeledTrainData.tsv', sep='\t')
print(data_train.shape)

texts = []
labels = []

for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx], "html.parser")
    texts.append(clean_str(text.get_text()))
    labels.append(data_train.sentiment[idx])

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Traing and validation set number of positive and negative reviews')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

GLOVE_DIR = "/home/shiva/work/textClassifier"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(
    len(word_index) + 1,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(
    LSTM(100, dropout=.3, recurrent_dropout=.3))(embedded_sequences)
preds = Dense(2, activation='softmax')(l_lstm)
model = Model(sequence_input, preds)
model.compile(
    loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

print("model fitting - Bidirectional LSTM")

model.summary()
model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    nb_epoch=10,
    batch_size=50)


# Attention GRU network
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1], ))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer,
              self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        # eij = K.tanh(K.dot(x, self.W))
        eij = K.tanh(K.dot(x, self.W))
        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


class AttentionWithContext(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 init='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            (input_shape[-1], 1),
            initializer=self.kernel_initializer,
            name='{}_W'.format(self.name),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.b = self.add_weight(
            (input_shape[1], ),
            initializer='zero',
            name='{}_b'.format(self.name),
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)

        self.u = self.add_weight(
            (input_shape[1], ),
            initializer=self.kernel_initializer,
            name='{}_u'.format(self.name),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.built = True

    def compute_mask(self, input, mask):
        return None

    def call(self, x, mask=None):
        # (x, 40, 300) x (300, 1)
        multData = K.dot(x, self.kernel)  # (x, 40, 1)
        multData = K.squeeze(multData, -1)  # (x, 40)
        multData = multData + self.b  # (x, 40) + (40,)

        multData = K.tanh(multData)  # (x, 40)

        multData = multData * self.u  # (x, 40) * (40, 1) => (x, 1)
        multData = K.exp(multData)  # (X, 1)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            mask = K.cast(mask, K.floatx())  #(x, 40)
            multData = mask * multData  #(x, 40) * (x, 40, )

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        multData /= K.cast(
            K.sum(multData, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        multData = K.expand_dims(multData)
        weighted_input = x * multData
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[-1], )


embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(
    len(word_index) + 1,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
# l_att = AttLayer()(l_gru)
l_att = AttentionWithContext()(l_gru)
preds = Dense(2, activation='softmax')(l_att)
model = Model(sequence_input, preds)
model.compile(
    loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

print("model fitting - attention GRU network")
model.summary()
model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    nb_epoch=10,
    batch_size=50)
