# Reddit_Toxicounter Progress Timeline
### 1/15/18
**All:** re-scrape data from 2005-2018 (randomly sample 100,000 comments per year)
**Minh:** revise Perspective API script
**Mikaela and Angel:** research models and write some ML scripts

**The research question(s) being answered:**
- How does toxicity change over time?
- How does toxicity change over different subreddits?
- What are the most toxic/least toxic subreddits?
- What are the most toxic/least toxic topics talked about?
- ?

**Models we can use to answer our research questions (supervised and unsupervised):**
- ?

### 1/8/18
We have redesigned our project goals and created a tentative progress timeline.

**The Project:** We will detect the toxicity of the discourse within the top 10 subreddits (based on Recent Activity on http://redditlist.com/ as of January 8, 2018) throughout 2015.

**Future Work:** We can build our own toxicity classifier for Reddit comments (instead of using Perspective API's models to label the data). We can also perform domain adaptation into Reddit comments after labeling a different dataset using either our model or the Perspective API.

**Progress Timeline:**

**Week 2-3:** "Ground Truth"
- Scrape all the data
  - Minh: The_Donald, videos, soccer
  - Mikaela: AskReddit, politics, worldnews, nba
  - Angel: todayilearned, funny, cfb
- Label our dataset for toxicity using Perspective API
- Clean all the data
  - Remove non-english characters
  - Convert html to their respective characters
  - Decide what to do with URLs, markdown formatting, emojis

**Week 4:** Model Design/Research
- TBA

### 12/16/18
We have finalized a method for scraping the reddit comment involving the Pushshift API. Here's the standard link that we will be working with
'https://api.pushshift.io/reddit/search/comment/?subreddit=' + subreddit + '&size=500&after='+ str(after_utc) + '&fields=parent_id,id,body,created_utc,score

The subreddit and after_utc variables need input. Furthermore, if we wanted to scrape comments in a certain timeframe, we can also add a before= variable to the url. For example, 'https://api.pushshift.io...size=500&after=0&before=.....'

Here is a good website to figure out utc - https://www.unixtimestamp.com/index.php

after_utc = scrape comments after this time
before_utc = scrape comments before this time

So if you want to scrape 2010-2013, after_utc = UNIX equivalent of 2010 and before_utc = UNIX equivalent of 2013

We have attached a new script to scrape reddit comments. The script was partially inspired by Sentdex on his "Building a Chatbot with Deep Learning' video series. The new script stores our comments in batches to a database file, so it does not run the risk of memory errors, etc. There is no need to install any modules because Python comes packaged with sqlite3. 

For first time users, please created a folder named "database" in the same directory as the script. Then, please input the subreddit name and after_utc variable at the top and run the script. The script will continually output the after_utc variable, so if you were to stop the script, make note of the last after_utc. Next time you run the script, you can input that number for the after_utc variable at the top and continue where you've left off. 

You can actually see what your databse looks like and what comments it contains if you download DB Browser of Sqlite. 
Link - https://sqlitebrowser.org/
Just go into the database folder, open up the reddit_comments.db file with DB Browser of Sqlite. 

### 11/27/18
Instead of scraping data using the Reddit API, we will now parse JSON data from https://files.pushshift.io/reddit/comments/. We will download all of the comments from 2010-2018 and store comments from the AskReddit, politics, The_Donald, worldnews, nba, videos, funny, todayilearned, soccer, and CFB subreddits (the top 10 SFW subreddits on http://redditlist.com/ based off of Recent Activity). Minh will download 2010-2013, Angel will download 2014-2016, Mikki will download 2017-2018, and each of us will run the script to collect data overnight.

Some other interesting things we can explore are tracking the toxicity levels and Reddit comment/subreddit content over time, and see how they coincide with real-world events.

**Tools we can use**
- JSON library for parsing
- Perspective API for labeling toxicity on our dataset
- Spacy similarity function/Word2Vec/TF-IDF

**Tasks to complete before the end of Winter Break**

Everyone:
- Decide which columns we want to use from dataset; make sure to check what columns are included in different years
- Add a date column to the dataset
- Gain access to Perspective API
- Exploratory Data Analysis
- Learn about modeling and decide which algorithms to use

Mikki and Angel:
- Test out reddit_comments_from_json.py on sample_data.json on our computers and update Minh (by the end of the week)
- Add toxicity labels to dataset using Perspective API (by the end of finals week)

Minh:
- Modeling (before/during Winter Break)


### 11/6/18
A lot of the top subreddits have primarily non-text based threads, albeit with many text comments. Thus, we will not remove non-text posts in our initial scrape of the data. Furthermore, we will be scraping the top 20 subreddits based on Recent Activity on http://redditlist.com/ as of **[insert date here]**.

What we did this meeting:
- Added the 'Submission Title' and 'Submission Selftext' columns to the comments.csv created by **reddit_scrape.py**
- Filtered out all the emojis and URLs

What we plan to do next meeting:
- Filter out non-English text and do other data cleaning
- Possibly explore Crimson Hexagon alternative to acquiring Reddit data
- See how Spacy deals with punctuation (look through old Comp Ling notes?)

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
