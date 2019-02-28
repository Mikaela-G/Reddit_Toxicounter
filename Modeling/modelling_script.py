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
    https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
    https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
    https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296
    https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
@author: Mikki
"""

# Miscellaneous
import sqlite3
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from collections import Counter
from gensim.models import Word2Vec
import warnings

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Keras
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Bidirectional, LSTM
from keras.layers.embeddings import Embedding

# ------------------------------------- Preprocessing ------------------------------------- #

# ensures reproducibility
np.random.seed(42)

# load database into dataframe
cnx = sqlite3.connect('/Users/Mikki/Documents/GitHub/Reddit_Toxicounter/database/reddit_comments_2.db')
df = pd.read_sql_query("SELECT * FROM AskReddit", cnx)

# use first 100 rows as dev set (COMMENT THIS OUT AFTER YOU FINISH SCRIPTING MODELS)
df = df[:200]

df = df.dropna()

# cleans text by removing characters and stop words
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwords from text
    return text

df['comment'] = df['comment'].apply(clean_text)
df['comment'] = df['comment'].replace(r'newlinechar', '', regex=True)

# averages the word vectors
def avg_word_vec(wordlist,size):
    # Otherwise return an average of the embeddings vectors
    sumvec=np.zeros(shape=(1,size))
    wordcnt=0
    for w in wordlist:
        if w in w2v_model:
            sumvec += w2v_model[w]
            wordcnt +=1
    if wordcnt ==0:
        return sumvec
    else:
        return sumvec / wordcnt

# tokenize comments
df['comment_tok'] = df['comment'].apply(lambda x: nltk.word_tokenize(x))

# count frequencies of each token
flat_list = [word for comment in df.comment_tok for word in comment]
c = Counter(flat_list)

# replace low frequency words (freq < 5) with <unk>
def low_freq_to_unk(comment):
    return [word if c[word]>5 else '<unk>' for word in comment]
df['comment_tok'] = df['comment_tok'].apply(low_freq_to_unk)

# split into train/test sets
X = df.comment_tok
y = df.toxic_label # NOTE: real_toxic_label for current data but TOXIC_LABEL FOR THE FINAL VERSION
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dimsize = 300

# preserving variables to be used for LSTM
x_train = X_train
x_test = X_test
Y_train = y_train
Y_test = y_test

# ------------------------------------- Word2Vec ------------------------------------- #

# train Word2Vec skip gram model on tokenized comments
w2v_model = Word2Vec(X_train, size=dimsize, window=10, workers=2, sg=1, negative=15, iter=5) # print(w2v_model)

# gets vocabulary
vocab = list(w2v_model.wv.vocab) # print(vocab)

# store word-vector pairs in dictionary
w2v_embed_dict = {word: w2v_model.wv[word] for word in vocab}

# saves word embeddings so they can be re-used for different classification models
#w2v_model.wv.save_word2vec_format('w2v_model_wv.txt', binary=False)

# ------------------------------------- Logistic Regression (Angel) ------------------------------------- #

# encoding the targets
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

#averages the word vectors for the comments
X_train=np.concatenate([avg_word_vec(w,dimsize) for w in X_train])
X_test=np.concatenate([avg_word_vec(w,dimsize) for w in X_test])

# Fitting Logistic Regression to the Training set
LR = LogisticRegression(random_state=0, solver='lbfgs',
                        multi_class='multinomial').fit(X_train, y_train)

#arguments used in metrics
y_true = y_test
y_predLR = LR.predict(X_test)

#metrics for determining success
print('LRaccuracy %s' % accuracy_score(y_true, y_predLR))
print(classification_report(y_true, y_predLR,))
print(confusion_matrix(y_true, y_predLR,))

# ------------------------------------- Naive Bayes (Angel) ------------------------------------- #

# Fitting Naive Bayes to the Training set
gnb = GaussianNB()
gnb.fit(X_train, y_train)

#metrics for determining success
y_predNB = gnb.predict(X_test)
print('NBaccuracy %s' % accuracy_score(y_true, y_predNB))
print(classification_report(y_true, y_predNB))
print(confusion_matrix(y_true, y_predNB))

# ------------------------------------- Random Forest (Angel) ------------------------------------- #

# Fitting Random Forest to the Training set
forest = RandomForestClassifier(n_estimators = 500)
forest.fit(X_train, y_train)

#metrics for determining success
y_predRF = forest.predict(X_test)
print('RFaccuracy %s' % accuracy_score(y_true, y_predRF))
print(classification_report(y_true, y_predRF))
print(confusion_matrix(y_true, y_predRF))

# ------------------------------------- SVM (Minh) ------------------------------------- #
# ------------------------------------- K-Nearest Neighbor (Mikki) ------------------------------------- #
# ------------------------------------- LSTM (Mikki) ------------------------------------- #

######### PREPROCESSING X_TRAINING DATA FOR MODEL #########

# fitting tokenizer on training data
t = Tokenizer()
t.fit_on_texts(x_train)
vocab_size = len(t.word_index) + 1 # t.word_index returns key-value pairs of token to unique integer

# integer encode tokenized comments (produces list of lists of tokens as integers)
x_train = t.texts_to_sequences(x_train)

# pad sequences so that each comment sequence is the same length
max_length = max([len(comment) for comment in x_train])
x_train = pad_sequences(x_train, maxlen=max_length, padding='post')

# create a weight matrix for training data words
embedding_matrix = np.zeros((vocab_size, dimsize))
for word, index in t.word_index.items():
    embed_vector = w2v_embed_dict.get(word) # matching words to their Word2Vec embeddings
    if embed_vector is not None:
        embedding_matrix[index] = embed_vector
    else:
        embedding_matrix[index] = w2v_embed_dict.get('<unk>') # unknown words are given the embedding for <unk>

######### PREPROCESSING X_TESTING DATA FOR MODEL #########

# fitting tokenizer on testing data
t = Tokenizer()
t.fit_on_texts(x_test)

# integer encode tokenized comments (produces list of lists of tokens as integers)
x_test = t.texts_to_sequences(x_test)

# pad sequences so that each comment sequence is the same length
x_test = pad_sequences(x_test, maxlen=max_length, padding='post')

######### PREPROCESSING Y_TARGET DATA FOR MODEL #########

# integer encoding and then one hot encoding toxicity labels
encoder = LabelEncoder()
encoder.fit_transform(y)
Y_train = encoder.transform(Y_train)
Y_test = encoder.transform(Y_test)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

######### MODELING #########

model = Sequential()
model.add(Embedding(vocab_size, dimsize, weights=[embedding_matrix], input_length=max_length))
model.add(Bidirectional(LSTM(dimsize)))
model.add(Dense(3, activation='softmax'))

# compile model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# summarize model
print(model.summary())

# fit the model
model.fit(x_train, Y_train, epochs=10, verbose=0)

# evaluate the model
loss, accuracy = model.evaluate(x_test, Y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))




# so when you build a machine learning model
# you take the words, tokenize them, map tokens to their vectors, get the sequence of vectors for a sentence

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
