#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:24:40 2019

@author: Mikki
"""
# first time users please run this SQL query in DBrowser for the database you are planning on labeling

# ALTER TABLE AskReddit
# ADD toxic_score REAL;
# ALTER TABLE AskReddit
# ADD severe_toxicity REAL;

import json
import requests
import sqlite3

api_token = "AIzaSyBma7dmCTo2Leiu56M5pWzhEA3CW_eu0Fk"
api_endpoint = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={}".format(api_token)
subreddit_name = "AskReddit" # this is your table's name. If my previous scripts worked it should just be the subreddit name
db_name = "reddit_comments_2" # this is the actual database's name, so check in the database folder for this

connection = sqlite3.connect("database/{}.db".format(db_name))
c = connection.cursor()

headers = {}
sql_transaction = []

def sql_update_1(toxic_score, comment_id):
    try:
        sql = "UPDATE '{}' SET toxic_score = {} WHERE comment_id = '{}';".format(subreddit_name, toxic_score, comment_id)
        transaction_bldr(sql)
    except Exception as e:
        print('SQL insertion',str(e))

def sql_update_2(severe_toxicity, comment_id):
    try:
        sql = "UPDATE '{}' SET severe_toxicity = {} WHERE comment_id = '{}';".format(subreddit_name, severe_toxicity, comment_id)
        transaction_bldr(sql)
    except Exception as e:
        print('SQL insertion',str(e))
        
def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) >= 10:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except Exception as e:
                print('SQL Transaction', str(e))
        connection.commit()
        sql_transaction = []

def main():
	c.execute('SELECT comment_id, comment FROM {}'.format(subreddit_name)) # WHERE unix > ___
	results = c.fetchall()
	for result in (comment for comment in results):
		payload = {
		  "comment": {
		     "text": "{}".format(result[1]),
		  },
		  "languages": ["en"],
		  "requestedAttributes": {
		    "TOXICITY": {},
             "SEVERE_TOXICITY": {}
		  }
		}
		r = requests.post(api_endpoint, data=json.dumps(payload), headers=headers)
		toxicity_score = json.loads(r.text)["attributeScores"]["TOXICITY"]['spanScores'][0]['score']['value']
		severe_toxicity = json.loads(r.text)["attributeScores"]["SEVERE_TOXICITY"]['spanScores'][0]['score']['value']
		sql_update_1(toxicity_score, result[0])
		sql_update_2(severe_toxicity, result[0])
        
if __name__ == '__main__':
	main()
