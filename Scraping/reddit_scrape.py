import praw
import pandas as pd
import re

# init
client_id = 'CLkv9Rhfadae3A'
client_secret = 'SAbAmTZxJzEYqVszkqqvWEVOS4M'

reddit = praw.Reddit(client_id = 'CLkv9Rhfadae3A',
                     client_secret = 'SAbAmTZxJzEYqVszkqqvWEVOS4M',
                     user_agent='Reddit_Toxicounter',
                     username='duyminh1998',
                     password='321thgifoN')

df = pd.DataFrame(columns = ['Submission Type', 'Submission ID', 'Submission Title', 'Submission Selftext'])

# displays the authorization status of our Reddit object
# print('Read only?: ', reddit.read_only)

# creates a Subreddit object for our desired subreddit
subreddit = reddit.subreddit('ukiyoe')

# prints the name and title of the subreddit to make sure it's correct
print('Subreddit name: ', subreddit.display_name)  # Output: redditdev
print('Subreddit title: ', subreddit.title)         # Output: reddit Development

# grabs all submissions ids for each submission types
hot_id = [submission.id for submission in subreddit.hot(limit=None)] 
hot_selftext = [submission.selftext for submission in subreddit.hot(limit=None)]
hot_title = [submission.title for submission in subreddit.hot(limit=None)]

j = 0

for i in range(len(hot_id)):
    df.loc[i, 'Submission Type'] = 'Hot'
    df.loc[i, 'Submission ID'] = hot_id[j]
    df.loc[i, 'Submission Title'] = hot_title[j]
    df.loc[i, 'Submission Selftext'] = hot_selftext[j]
    j += 1

new_id = [submission.id for submission in subreddit.new(limit=None)] 
new_selftext = [submission.selftext for submission in subreddit.new(limit=None)]
new_title = [submission.title for submission in subreddit.new(limit=None)]

k = 0

for i in range(len(new_id)):
    df.loc[i, 'Submission Type'] = 'new'
    df.loc[i, 'Submission ID'] = new_id[k]
    df.loc[i, 'Submission Title'] = new_title[k]
    df.loc[i, 'Submission Selftext'] = new_selftext[k]
    k += 1

controversial_id = [submission.id for submission in subreddit.controversial(limit=None)] 
controversial_selftext = [submission.selftext for submission in subreddit.controversial(limit=None)]
controversial_title = [submission.title for submission in subreddit.controversial(limit=None)]

l = 0

for i in range(len(controversial_id)):
    df.loc[i, 'Submission Type'] = 'controversial'
    df.loc[i, 'Submission ID'] = controversial_id[l]
    df.loc[i, 'Submission Title'] = controversial_title[l]
    df.loc[i, 'Submission Selftext'] = controversial_selftext[l]
    l += 1

top_id = [submission.id for submission in subreddit.top(limit=None)]
top_selftext = [submission.selftext for submission in subreddit.top(limit=None)] 
top_title = [submission.title for submission in subreddit.top(limit=None)]

h = 0

for i in range(len(top_id)):
    df.loc[i, 'Submission Type'] = 'top'
    df.loc[i, 'Submission ID'] = top_id[h]
    df.loc[i, 'Submission Title'] = top_title[h]
    df.loc[i, 'Submission Selftext'] = top_selftext[h]
    h += 1

rising_id = [submission.id for submission in subreddit.rising(limit=None)]
rising_selftext = [submission.selftext for submission in subreddit.rising(limit=None)] 
rising_title = [submission.title for submission in subreddit.rising(limit=None)]

w = 0

for i in range(len(rising_id)):
    df.loc[i, 'Submission Type'] = 'rising'
    df.loc[i, 'Submission ID'] = rising_id[w]
    df.loc[i, 'Submission Title'] = rising_title[w]
    df.loc[i, 'Submission Selftext'] = rising_selftext[w]
    w += 1

# drops any duplicates
df = df.drop_duplicates()

# saves to csv
df.to_csv('submission_id.csv')

# filter and saves comments
df = pd.read_csv('submission_id.csv')

comment_df = pd.DataFrame(columns = ['Submission ID', 'Comments'])
q = 0

for post_id in df['Submission ID']:
    submission = reddit.submission(id = post_id)
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        comment_df.loc[q, 'Submission ID'] = post_id

        # filter comment string
        encoded = comment.body.encode("utf-8", errors='ignore') # makes dealing with emojis possible
        edited_comment = re.sub(r'http\S+', '', str(encoded)) # ignores URLs
        edited_comment = re.sub(r'\\\w*', ' ', edited_comment) # ignores emojis
        comment_df.loc[q, 'Comments'] = edited_comment.strip()
        q += 1

comment_df.to_csv('comments.csv')
