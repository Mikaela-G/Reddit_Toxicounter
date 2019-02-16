# Project Page Template (Header)

## Abstract

## Contributors 
* Mikaela Guerrero 
* Minh Hua
* Angel Chavez

## Motivation

## Scraping Methodology
Important in every data analysis task is the retrieval, wrangling, and cleaning of data - in this respect, our group had no shortage of challenges. The number one concern when dealing with Reddit comment data is its sheer volume - one month's worth of comments from one subreddit can take more than 5 GB to store. This roadblock generated two practical concerns: the time it takes to scrape the data, and the memory limitations Python imposes. As a result, clever data scraping and handling technique had to be implemented in order to surmount the obstacle. 

In the beginning, we used the Praw API to directly scrape Reddit for comments, but this endeavour was hampered by shoddy internet connections and Python's memory limitations. As a result, we examined Pushshift.io, an online database/archive of **all** Reddit comments from the site's conception. The site had an API that allowed for intelligent queries of comments from specific subreddits, timeframes, alongside other criterion. 

At our project's conception, we weren't clear and concise on our thesis question, which lead to a scraping pattern that was highly inefficient; we basically scraped comments from ten of the most popular subreddits, which soon proved to be impossible due to volume. As a result, once we improved upon our research question to only scrape **one** subreddit, we were able to much more efficiently retrieve the data. 

This also brought with it many challenges. As mentioned before, one month's worth of comments from one subreddit alone can tally up to 20 million comments. Pressured by an approaching deadline and limitations on work hours, we came to the conclusion that scraping samples of data were the optimal solution. By scraping a predetermined percentage of comments from each month, we were able to cover the scope of a subreddit's sentiment, to see how its temperament change over time, how it reacts to noteworthy events. We finally reached the best scraping mindset, and all that was left to do was implemenation. 

In terms of implementation, the workhorse of our scripts were requests, aiohttp, asyncio, and sqlite3. These four "horsemen of the pycalypse" ran incessantly for weeks, retrieving thousands of comments per second and storing them in a local SQL database. Here's a sample pseudo code of the process:

```python
subreddit = "AskReddit"
timeframes = [(1233446400, 1235865600), (1235865600, 1238544000), (1238544000, 1272672000), (1272672000, 1243814400), (1243814400, 1246406400), (1246406400, 1249084800), (1249084800, 1251763200), (1251763200, 1254355200), (1254355200, 1257033600), (1257033600, 1259625600), (1259625600, 1262304000)]
sql_transaction = []

connection = sqlite3.connect('database/{}.db'.format(subreddit + timeframe))
c = connection.cursor()

def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS " + subreddit + " (comment_id TEXT PRIMARY KEY, comment TEXT, unix INT, score INT, toxicity REAL)")

def clean_data(data):
    comment = data.replace('\n',' newlinechar ').replace('\r',' newlinechar ').replace('"',"'")
    # comment = re.sub(r'http\S+', '', comment)
    comment = re.sub('&gt;', '', comment)
    comment = re.sub('&lt;', '', comment)
    return comment

def sql_insert(commentid, comment, time, score):
    try:
        sql = """INSERT INTO """ + subreddit + """(comment_id, comment, unix, score) VALUES ("{}","{}",{},{});""".format(commentid, comment, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('SQL insertion',str(e))

if __name__ == '__main__':
    try:
    	for frame in timeframes:
            after_utc = frame[0]
            before_utc = frame[1]
            create_table()
            continue_scrape = True
            while number_processed < 250:
                r = requests.get('https://api.pushshift.io/reddit/search/comment/?subreddit={}&size=500&after={}&before={}&fields=id,body,created_utc,score'.format(subreddit, str(after_utc), str(before_utc)))
                parsed_json = json.loads(r.text)
                if len(parsed_json['data']) > 0:
                        comment_processed = 0
                        for comment in parsed_json['data']:
                            body = clean_data(comment['body'])
                            if acceptable(body):
                                created_utc = comment['created_utc']
                                score = comment['score']
                                comment_id = comment['id']
                                sql_insert(comment_id, body, created_utc, score)
                                comment_processed += 1
                            else:
                                pass
                        c.execute("VACUUM")
                        connection.commit()
                        after_utc = parsed_json['data'][-1]['created_utc']
                        number_processed += 1
                        print('Number of pages processed: {}'.format(number_processed), 'Number of comments processed: {}'.format(comment_processed), 'Current UTC: {}'.format(after_utc))
```
And here is a breakdown of the process:
1. Create a database
```python 
connection = sqlite3.connect('database/{}.db'.format(subreddit + timeframe))
c = connection.cursor()

def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS " + subreddit + " (comment_id TEXT PRIMARY KEY, comment TEXT, unix INT, score INT, toxicity REAL)")
```
2. Scrape data using requests, pushshift.io
```python 
r = requests.get('https://api.pushshift.io/reddit/search/comment/?subreddit={}&size=500&after={}&before={}&fields=id,body,created_utc,score'.format(subreddit, str(after_utc), str(before_utc))
```
3. Extract the data from the json
```python
parsed_json = json.loads(r.text)
```
4. Insert data into the database
```python
sql_insert(comment_id, body, created_utc, score)
```
5. Rinse and repeat

## Analysis Methodology

## Results

# Reddit_Toxicounter Progress Timeline
### 1/15/18
**Minh:**
- re-scrape and label AskReddit comments from 2005-2018 (randomly sample 20% for each year?)

**Mikaela and Angel:**
- research models and write some ML scripts

**The research question(s) being answered:**
- How does toxicity change over time?
   - Most toxic timeframes/dates?
- What are the most toxic/least toxic topics talked about?
- How is the number of upvotes/downvotes correlated with toxicity?
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
