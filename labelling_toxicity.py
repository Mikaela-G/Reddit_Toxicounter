#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:24:40 2019

@author: Mikki

Purpose: Making a POST request to the Perspective API to rate toxicity for each comment
"""
import json
import requests

api_token = INSERT_API_TOKEN_HERE
api_endpoint = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

# for each comment in the SQLite database file
# data to be sent to API
comment = {"comment.text": INSERT_COMMENT_HERE, "requestedAttributes": INSERT_MODEL_HERE, "languages": "en"}

# sending post request and saving request-response as a response object
r = requests.post(url = api_endpoint, data = comment)

# extracting content of the response in unicode
toxicity_score = r.text