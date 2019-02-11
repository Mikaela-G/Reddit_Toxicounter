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
#import sklearn.model_selection
#from sklearn.preprocessing import LabelEncoder
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
pd.set_option('display.max_columns', None)


# load database into dataframe ##change the path and subreddit accordingly
cnx = sqlite3.connect('C:\\Users\\angel\\OneDrive\\Documents\\GitHub\\Reddit_Toxicounter\\database\\reddit_comments.db')
df = pd.read_sql_query("SELECT * FROM todayilearned", cnx)


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
    #text = text.lower() # lowercase text
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
#df['comment'].apply(lambda x: len(x.split(' '))).sum()

# tokenize comments
df['comment_tok'] = df['comment'].apply(lambda x: nltk.word_tokenize(x))

# ------------------------------------- Logistic Regression ------------------------------------- #
# split into train/test sets
X = df.comment_tok
y = df.toxic_label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

dimsize = 300 #size of hidden layer
w2v_model = Word2Vec(X_train, size=dimsize, window=10, min_count=10, negative=15, iter=10, sg=1) # skip gram

# gets vocabulary
vocab = list(w2v_model.wv.vocab)

#averages the word vectors for the comments
X_train=np.concatenate([avg_word_vec(w,dimsize) for w in X_train])
X_test=np.concatenate([avg_word_vec(w,dimsize) for w in X_test])

#trains the model with the training data
clf = linear_model.SGDClassifier(loss='log')
clf.fit(X_train, y_train)

#arguments used in 
y_true = y_test
y_pred = clf.predict(X_test)

#metrics for determining success
print('accuracy %s' % accuracy_score(y_true, y_pred))

toxic_labels = ['not toxic', 'moderately toxic', 'very toxic']
print(classification_report(y_true, y_pred,target_names=toxic_labels))
print(confusion_matrix(y_true, y_pred, toxic_labels))
