# README - Reddit Toxicounter

## Abstract
This project seeks to classify speech toxicity through the use of machine learning models. Our toxicity ratings are derived from the [Perspective API](https://www.perspectiveapi.com/#/) and our models were built in Python with the aid of libraries such as [scikit-learn](https://scikit-learn.org/stable/) and [Keras](https://keras.io/). We gathered [AskReddit](https://www.reddit.com/r/AskReddit/) comments from 2005 to 2018 from [pushshift.io](https://pushshift.io/) and stored them in [SQLite3](https://www.sqlite.org/index.html) databases. [insert stuff about visualizations here].

## Contributors 
* Mikaela Guerrero 
* Minh Hua
* Angel Chavez

## Motivation
The internet has no shortage of toxicity, including threats, derogatory comments, and other such hate speech. For example, in less than 24 hours, Twitter users unwittingly managed to force Microsoft to shut down its chatbot Tay, by teaching it to repeat extremely hate-filled discourse. As such, large social media sites (such as Reddit!) are very hard to moderate.

Before:

![](https://confluo.files.wordpress.com/2016/03/yyyy.png?w=640)

After:

![](https://i.imgur.com/L2JRI7r.png)

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
Our analysis on the toxicity of Reddit comments deals with a relatively standard machine learning problem: multi-label classification. We deal with supervised learning, which in this instance, involves training a classifier on data already labeled for toxicity in order to predict the toxicity level of other comments.

Using Perspective API's toxicity and severe toxicity models, we labeled our data with percentages that score toxicity for any given comment. We then created our own ground truth by placing comments into one of three categories (not toxic, moderately toxic, and very toxic) based on Perspective's ratings.

Our labels were created with the following thresholds:
    *
    *
    *

## Our Text Classification Pipeline
![pipeline](https://i.imgur.com/iuGu6RD.png)

## From Data to Word2Vec
Machine learning models typically take numerical features as input. Since our Reddit data is text-based, we needed some way to transform the comments into numerical input readable by the machine learning models. Thus, we generated word embeddings from the tokenized Reddit comments using a Word2Vec skip-gram model. Word embeddings are vectorized representations of words mapped to the same vector space and positioned according to similarity. Skip-gram architecture involves taking a single word and attempting to predict words that might occur alongside the target word. We used Word2Vec's skip-gram rather than CBOW (Continuous Bag of Words) since skip-gram deals better with infrequent words.

## From Word2Vec to Machine Learning Models
We then mapped each of the words in our dataset to their respective embeddings, with the intent of using said embeddings as feature vectors for supervised learning. We developed the following classification models:

    * Logistic Regression
    * Naive Bayes
    * Random Forest
    * LSTM

Each of these models took word embeddings as the input and predicted toxicity labels as the output.
    
## Logistic Regression

## Naive Bayes

## Random Forest

## LSTM (Long Short Term Memory)

# REFORMAT THIS WHOLE SECTION (IGNORE THE BELOW)
## A Brief Overview of Classification
* Two variables considered in building ML models:
    * Comment (tokenized and mapped to word embeddings)
    * Toxicity label for each comment
        * Very toxic; had a score between [aaa and aaa] for toxicity, [aaa and aaa] for severe toxicity
        * Moderately toxic; had a score between [aaa and aaa] for toxicity, [aaa and aaa] for severe toxicity
        * Not toxic; had a score between [aaa and aaa] for toxicity, [aaa and aaa] for severe toxicity

## Results

## Future Work
