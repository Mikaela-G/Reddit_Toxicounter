# Reddit_Toxicounter Progress Timeline
### 11/6/18
A lot of the top subreddits have primarily non-text based threads, albeit with many text comments. Thus, we will not remove non-text posts in our initial scrape of the data. Furthermore, we will be scraping the top 20 subreddits based on Recent Activity on http://redditlist.com/ as of [insert date here].

What we did this meeting:

What we plan to do next meeting:

### 10/30/18
In the pursuit of scraping cleaner data from the get-go, we might want to leave out non-text posts. The following link describes how to implement this in **reddit_scrape.py**.
https://www.reddit.com/r/redditdev/comments/9064t0/praw_how_can_i_return_what_type_of_post_a/

What we did this meeting:
- Added the 'Submission Type' column to the comments.csv created by **reddit_scrape.py**

What we plan to do next meeting:
- Decide on which subreddits/how many subreddits we're going to scrape (for now at least)
- Discuss whether or not to leave in non-text posts (with and without comments)

### 10/23/18
Potential methods of scraping the data:
- using reddit api
- ~~using requests & beautiful soup to get data in html format~~
- ~~using json links to get data in json format~~

Potential roadblocks:
- using reddit api; can't scrape all the thread titles? (stops after a certain point)
- figuring out how to get the comments

PRAW Documentation to Read:
- Working with PRAW's Models: Submission
- Working with PRAW's Models: Subreddit
- Comment Extraction and Parsing

What we plan to do before next meeting:
- read the docs
- scrape all threads of the subreddits, scrape all comments and replies
- brainstorm subreddits, number of subreddits
- try to save data into csv
