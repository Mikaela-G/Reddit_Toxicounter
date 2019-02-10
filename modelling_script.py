#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:18:13 2019
Goals:
    Generate word embeddings using Word2Vec's Skip-Gram Model
    Create multi-class classification models to predict toxic_label from comment vectors
        Logistic Regression (Angel)
        Naive Bayes (Angel)
        Random Forest (Angel)
        SVM (Mikki)
        K-Nearest Neighbor (Minh)
        LSTM (Mikki)
        BERT? using AWS server
References:
    https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
    https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296
    https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
@author: Mikki
"""

# Here are the changes we're going to make to the structure of our script.
# * COMPLETE * 1) We're going to take out the lower 1/3 or 2/3 frequent words and replace them with <UNK>
# * COMPLETE * 2) We're going to train the Word2Vec model on the entire dataframe, not just on the train set.
#   (In other words, we'll train/test split after creating the w2v model)
# * COMPLETE * 3) When we do the train/test split, we'll set the seed so that it's replicable for all the models (since it's usually randomized).
# 4) For logistic regression, (and possibly other sklearn algorithms) we'll want to take either the min, max, or mean of the word vectors within each sentence instead of feeding the embeddings one by one like we'll do for the RNN.
# * COMPLETE * 5) When developing your models, use a very small subset of the data (only 50 rows) so that it runs faster!

import sqlite3
import pandas as pd
import nltk
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

# ------------------------------------- Preprocessing ------------------------------------- #

# load database into dataframe
cnx = sqlite3.connect('/Users/Mikki/Documents/GitHub/Reddit_Toxicounter/database/reddit_comments_2.db')
df = pd.read_sql_query("SELECT * FROM AskReddit", cnx)

# use first 50 rows as dev set (COMMENT THIS OUT AFTER YOU FINISH SCRIPTING MODELS)
df = df[:50]

# strips out any weird characters and newlinechars
df['comment'] = df['comment'].replace('[^a-zA-Z0-9.,;:/\[\]\(\) ]', '', regex=True)
df['comment'] = df['comment'].replace(r'newlinechar', '', regex=True)

# tokenize comments
df['comment_tok'] = df['comment'].apply(lambda x: nltk.word_tokenize(x))

# count frequencies of each token
flat_list = [word for comment in df.comment_tok for word in comment]
c = Counter(flat_list)

# replace low frequency words (freq < 5) with <UNK>
def low_freq_to_UNK(comment):
    return [word if c[word]>5 else '<UNK>' for word in comment]
df['comment_tok'] = df['comment_tok'].apply(low_freq_to_UNK)

# ------------------------------------- Word2Vec ------------------------------------- #

# train Word2Vec skip gram model on tokenized comments
w2v_model = Word2Vec(df.comment_tok, size=300, window=10, min_count=5, negative=15, iter=10, sg=1) # print(w2v_model)

# gets vocabulary
vocab = list(w2v_model.wv.vocab) # print(words)

# store word-vector pairs in dictionary
w2v_embed_dict = {word: w2v_model.wv[word] for word in vocab}

# saves word embeddings so they can be re-used for different classification models
#w2v_model.wv.save_word2vec_format('w2v_model_wv.txt', binary=False)

# ------------------------------------- Logistic Regression (Angel) ------------------------------------- #

# split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(df['comment_tok'], df['real_toxic_label'], test_size=0.2, random_state=42)

# ------------------------------------- Naive Bayes (Angel) ------------------------------------- #
# ------------------------------------- Random Forest (Angel) ------------------------------------- #
# ------------------------------------- SVM (Mikki) ------------------------------------- #
# ------------------------------------- K-Nearest Neighbor (Minh) ------------------------------------- #
# ------------------------------------- LSTM (Mikki) ------------------------------------- #

# define documents
docs = list(df.comment)

# encode and define labels as numbers --> have to do one hot encoding!!!
def encode_toxic_label(label):
    if label == 'not toxic':
        return 0
    elif label == 'moderately toxic':
        return 1
    elif label == 'very toxic':
        return 2
df['toxic_label_num'] = df['real_toxic_label'].apply(encode_toxic_label)
labels = np.array(list(df.toxic_label_num))

# prepare tokenizer (t.word_index returns key-value pairs of token to unique integer)
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1

# integer encode documents (produces list of lists of tokens as integers)
encoded_docs = t.texts_to_sequences(docs)

# pad sequences so that each doc is the same length
max_length = max([len(doc) for doc in encoded_docs])
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 300))
for word, index in t.word_index.items():
    embed_vector = w2v_embed_dict.get(word)
    if embed_vector is not None:
        embedding_matrix[index] = embed_vector

# define model
model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=max_length))
model.add(Flatten())
model.add(Dense(10000, activation='relu'))
model.add(Dense(100, activation='softmax'))

# compile model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# summarize model
print(model.summary())

# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)

# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))




# so when you build a machine learning model
# you take the words
# tokenize them
# then you map tokens to their vectors
# then you get the sequence of vectors for a sentence

# you would feed in each embedding one by one
# so if your comment was "I like pie"
# you'd use the embeddings of "I", "like" and "pie" as your input





# we're making two types of models:
#   3 part classification models that predict toxicity label from word embeddings (and nothing else)
#       3 potential labels:
#           very toxic -> >90% toxic_score and >70% severe_toxicity
#           moderately toxic -> <90% & >50% toxic_score and <70% & >50% severe_toxicity
#           not toxic -> <50% toxic_score
#           (see if there's any cases where there's <50% toxic_score but it's also severely toxic)
#       algorithms we can use:
#           Naive Bayes Classifier 
#           Logistic Regression
#           Random Forest
#           SVM
#           Recurrent Neural Network
#           Nearest Neighbor
#   regression models that analyze the interaction between toxic_label, time, and score

# things we need to do before modeling:
#   create 2 new columns: severe_toxicity, toxic_label
#       (toxicity_label will be our new ground truth)
#   create word embeddings using Word2Vec

# for the first type of model:
#   input for the main model = word embeddings; other linguistic information (like number of uppercase or punctuation or etc.)
#   output for the main model = toxicity label
# for the second type of model:
#   exploring the relationships between the predictors and dependent variable
#   dependent variable can be toxicity_label, if we want to see how time and score impact toxicity

# stuff for visualization rather than modeling:
#   relationship between time and toxicity_label
#   relationship between time and score
