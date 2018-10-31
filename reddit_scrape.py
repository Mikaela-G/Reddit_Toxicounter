import praw
import pandas as pd

# init
client_id = 'CLkv9Rhfadae3A'
client_secret = 'SAbAmTZxJzEYqVszkqqvWEVOS4M'

reddit = praw.Reddit(client_id = 'CLkv9Rhfadae3A',
                     client_secret = 'SAbAmTZxJzEYqVszkqqvWEVOS4M',
                     user_agent='Reddit_Toxicounter',
                     username='duyminh1998',
                     password='321thgifoN')

df = pd.DataFrame(columns = ['Submission Type', 'Submission ID'])

# displays the authorization status of our Reddit object
# print('Read only?: ', reddit.read_only)

# creates a Subreddit object for our desired subreddit
subreddit = reddit.subreddit('TheDonald')

# prints the name and title of the subreddit to make sure it's correct
print('Subreddit name: ', subreddit.display_name)  # Output: redditdev
print('Subreddit title: ', subreddit.title)         # Output: reddit Development

# generates Subreddit objects for submission types
hot_subreddit = subreddit.hot(limit=None)
new_subreddit = subreddit.new(limit=None)
controversial_subreddit = subreddit.controversial(limit=None)
top_subreddit = subreddit.top(limit=None)
# rising_subreddit = subreddit.rising(limit=None)

# grabs all submissions ids for each submission types
hot_id = [submission.id for submission in hot_subreddit]

j = 0

for i in range(len(hot_id)):
    df.loc[i, 'Submission Type'] = 'Hot'
    df.loc[i, 'Submission ID'] = hot_id[j]
    j += 1

new_id = [submission.id for submission in new_subreddit]

k = 0

for i in range(len(hot_id), len(hot_id) + len(new_id)):
    df.loc[i, 'Submission Type'] = 'new'
    df.loc[i, 'Submission ID'] = new_id[k]
    k += 1

controversial_id = [submission.id for submission in controversial_subreddit]

l = 0

for i in range(len(hot_id) + len(new_id), len(hot_id) + len(new_id) + len(controversial_id)):
    df.loc[i, 'Submission Type'] = 'controversial'
    df.loc[i, 'Submission ID'] = controversial_id[l]
    l += 1

top_id = [submission.id for submission in top_subreddit]

h = 0

for i in range(len(hot_id) + len(new_id) + len(controversial_id), len(hot_id) + len(new_id) + len(controversial_id) + len(top_id)):
    df.loc[i, 'Submission Type'] = 'top'
    df.loc[i, 'Submission ID'] = top_id[h]
    h += 1

'''rising_id = [submission.id for submission in rising_subreddit]

w = 0

for i in range(len(hot_id) + len(new_id) + len(controversial_id) + len(top_id), len(hot_id) + len(new_id) + len(controversial_id) + len(top_id) + len(rising_id)):
    df.loc[i, 'Submission Type'] = 'rising'
    df.loc[i, 'Submission ID'] = rising_id[w]
    w += 1'''

# drops any duplicates
df = df.drop_duplicates()

# saves to csv
df.to_csv('submission_id.csv')


# comments take a LONG time to scrape so we will not do it till we're ready
'''df = pd.read_csv('submission_id.csv')

comment_df = pd.DataFrame(columns = ['Submission Type', 'Submission ID', 'Comments'])
q = 0

for post_id in df['Submission ID']:
    submission = reddit.submission(id = post_id)
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        comment_df.loc[q, 'Submission Type'] = df.iloc[q]['Submission Type']
        comment_df.loc[q, 'Submission ID'] = post_id
        comment_df.loc[q, 'Comments'] = comment.body
        q += 1

comment_df.to_csv('comments.csv')'''
