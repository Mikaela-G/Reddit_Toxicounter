import json
import pandas as pd

filename = 'RC_2005-12' # input your file name here
subreddit_list = ['AskReddit', 'politics', 'The_Donald', 'worldnews', 'nba', 'videos', 'funny', 'todayilearned', 'soccer', 'CFB'] 

comments = []

for line in open(filename, 'r'):
    comments.append(json.loads(line))

dictkeys = ['controversiality', 'subreddit', 'score', 'body', 'permalink']

df = pd.DataFrame(columns = dictkeys)
j = 0
for subreddit_name in subreddit_list:
    for i in range(len(comments)):
        if comments[i]['subreddit'] == subreddit_name:
            for key in dictkeys:
                df.at[j, key] = comments[i][key]
            j += 1
        else:
            pass

df.to_csv(filename + '.csv') # the output is your subreddit's name + csv
