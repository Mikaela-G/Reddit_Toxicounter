#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:18:13 2019
Goal: Generate word embeddings using Word2Vec's Skip-Gram Model
Reference 1: https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
Reference 2: https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296
@author: Mikki
"""

import sqlite3
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

# ------------------------------------- Word2Vec ------------------------------------- #

# load database into dataframe
cnx = sqlite3.connect('/Users/Mikki/Documents/GitHub/Reddit_Toxicounter/database/reddit_comments_2.db')
df = pd.read_sql_query("SELECT * FROM AskReddit", cnx)

# tokenize comments
df['comment_tok'] = df['comment'].apply(lambda x: nltk.word_tokenize(x))

# split into train/test sets
train, test = train_test_split(df, test_size=0.2)

# train Word2Vec model on tokenized comments
w2v_model = Word2Vec(train.comment_tok, size=300, window=10, min_count=5, negative=15, iter=10, sg=1) # skip gram
print(w2v_model)

# gets vocabulary
words = list(w2v_model.wv.vocab)
print(words)

# saves word embeddings so they can be re-used for different classification models
w2v_model.wv.save_word2vec_format('w2v_model_wv.txt', binary=False)

# ------------------------------------- Logistic Regression ------------------------------------- #

# ------------------------------------- Random Forest ------------------------------------- #

# ------------------------------------- SVM ------------------------------------- #

# ------------------------------------- RNN? ------------------------------------- #





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
