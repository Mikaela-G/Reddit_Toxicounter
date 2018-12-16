import sqlite3
import json
import requests

sql_transaction = []

connection = sqlite3.connect('database/{}.db'.format('reddit_comments'))
subreddit = 'The_Donald' # input the subreddit name here
after_utc = 1469073941 # if this is your first time scraping, put 0, afterwards, whenever you stop scraping, make note of the UTC of the last comment, and put it here next time you scrape
#before_utc = if you use this variable, please also modify the url link
continue_scrape = True
c = connection.cursor()

def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS " + subreddit + " (parent_id TEXT, comment_id TEXT PRIMARY KEY, comment TEXT, unix INT, score INT)")

# filter comment
def clean_data(data):
    data = data.replace('\n',' newlinechar ').replace('\r',' newlinechar ').replace('"',"'")
    return data

def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 500:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except Exception as e:
                print('SQL Transaction', str(e))
        connection.commit()
        sql_transaction = []

def acceptable(data):
    if len(data) > 32000:
        return False
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    else:
        return True

def sql_insert(parentid, commentid, comment, time, score):
    try:
        sql = """INSERT INTO """ + subreddit + """(parent_id, comment_id, comment, unix, score) VALUES ("{}","{}","{}",{},{});""".format(parentid, commentid, comment, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('SQL insertion',str(e))

if __name__ == '__main__':
    create_table()
    while continue_scrape:
        try:
            r = requests.get('https://api.pushshift.io/reddit/search/comment/?subreddit=' + subreddit + '&size=500&after='+ str(after_utc) + '&fields=parent_id,id,body,created_utc,score')
            parsed_json = json.loads(r.text)
            if len(parsed_json['data']) > 0:
                    for comment in parsed_json['data']:
                        body = clean_data(comment['body'])
                        if acceptable(body):
                            created_utc = comment['created_utc']
                            score = comment['score']
                            comment_id = comment['id']
                            parent_id = comment['parent_id']
                            sql_insert(parent_id, comment_id, body, created_utc, score)
                        else:
                            pass
                    after_utc = parsed_json['data'][-1]['created_utc']
            else:
                    continue_scrape = False
            print(after_utc) 
        except Exception as e:
            print(str(e))
