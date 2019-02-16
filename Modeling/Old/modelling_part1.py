# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 23:41:19 2019

@author: angel
"""
import sqlite3
import pandas as pd
import numpy as np
import re
import nltk
from gensim.models import Word2Vec
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
import warnings
#import sklearn.model_selection
#from sklearn.preprocessing import LabelEncoder
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")
# load database into dataframe ##change the path and subreddit accordingly
cnx = sqlite3.connect('C:\\Users\\angel\\OneDrive\\Documents\\GitHub\\Reddit_Toxicounter\\database\\AskReddit_2008 - 2011.db')
df1 = pd.read_sql_query("SELECT * FROM AskReddit", cnx)
df  = df1.head(210000)
df = df.dropna()
#CLEANS THE TEXT BY REMOVING CHARACTERS AND STOPWORDS
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

#averages the word vectors
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
df['comment'] = df['comment'].apply(clean_text)

# tokenize comments
df['comment_tok'] = df['comment'].apply(lambda x: nltk.word_tokenize(x))

# ---------------------------------------- Set Up ---------------------------------------- #
#split into train/test sets
X = df.comment_tok
y = df.toxic_label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
#encoding the targets
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
dimsize = 300 #size of hidden layer
#Train the Word2Vec model on the training data
w2v_model = Word2Vec(X_train, size=dimsize, window=10, min_count=10, negative=15, iter=10, sg=1) # skip gram
vocab = list(w2v_model.wv.vocab) # gets vocabulary

#averages the word vectors for the comments
X_train=np.concatenate([avg_word_vec(w,dimsize) for w in X_train])
X_test=np.concatenate([avg_word_vec(w,dimsize) for w in X_test])

# ---------------------------------------- Logistic Regression ---------------------------------------- #
# Fitting Logistic Regression to the Training set
LR = linear_model.SGDClassifier(loss='log')
LR.fit(X_train, y_train)

#arguments used in metrics
y_true = y_test
y_predLR = LR.predict(X_test)

#metrics for determining success
print('LRaccuracy %s' % accuracy_score(y_true, y_predLR))
print(classification_report(y_true, y_predLR,))
print(confusion_matrix(y_true, y_predLR,))

# ---------------------------------------- Naive Bayes ---------------------------------------- #
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

#metrics for determining success
y_predNB = gnb.predict(X_test)
print('NBaccuracy %s' % accuracy_score(y_true, y_predNB))
print(classification_report(y_true, y_predNB))
print(confusion_matrix(y_true, y_predNB))

# ---------------------------------------- Random Forest --------------------------------------------- #
# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 500)
forest.fit(X_train, y_train)

#metrics for determining success
y_predRF = forest.predict(X_test)
print('RFaccuracy %s' % accuracy_score(y_true, y_predRF))
print(classification_report(y_true, y_predRF))
print(confusion_matrix(y_true, y_predRF))