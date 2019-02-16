import asyncio
from aiohttp import ClientSession
import sqlite3
import json
import re

conn = sqlite3.connect("database/AskReddit_2008 - 2011.db")
c = conn.cursor()

headers = {}
api_token = "AIzaSyBma7dmCTo2Leiu56M5pWzhEA3CW_eu0Fk"
api_endpoint = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={}".format(api_token)

c.execute("SELECT unix from AskReddit WHERE toxicity IS NOT NULL ORDER BY unix DESC LIMIT 1;")
last_utc = c.fetchall()[0][0]
sql_transaction = []

continue_label = True

async def fetch(url, session, data, headers):
    async with session.post(url, data=json.dumps(data), headers=headers) as response:
        return await response.read()
    
def sql_update(toxic_score, severe_toxic, comment_id):
    try:
        sql = "UPDATE AskReddit SET toxicity = {}, severe_toxic = {} WHERE comment_id = '{}';".format(toxic_score, severe_toxic, comment_id)
        transaction_bldr(sql)
    except Exception as e:
        print('SQL insertion',str(e))

def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 449:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except Exception as e:
                print('SQL Transaction', str(e))
        conn.commit()
        sql_transaction = []

def clean_comment(data):
    comment = data.replace(' newlinechar ', '').replace('"',"'")
    # comment = re.sub(r'http\S+', '', comment)
    comment = re.sub('&amp', '', comment)
    return comment

async def run(api_endpoint, payloads, headers, comment_ids):
    tasks = []
    async with ClientSession() as session:
        for payload in payloads:
            task = asyncio.ensure_future(fetch(api_endpoint, session, payload, headers))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        print("Number of Responses: ", len(responses))
        for response, comment_id in zip(responses, comment_ids):
            try:
                toxicity = json.loads(response.decode('utf-8'))["attributeScores"]["TOXICITY"]['spanScores'][0]['score']['value']
                severe_toxic = json.loads(response.decode('utf-8'))["attributeScores"]["SEVERE_TOXICITY"]['spanScores'][0]['score']['value']
                sql_update(toxicity, severe_toxic, comment_id)
            except Exception as e:
                print(str(e))
                pass

if __name__ == '__main__':
    i = 0
    while continue_label == True:
        try:
            payloads = []
            c.execute('SELECT comment_id, comment, unix FROM {} WHERE unix > {} LIMIT 450'.format("AskReddit", last_utc))
            results = c.fetchall()
            if len(results) > 0:
                for result in (comment for comment in results):
                    if result[1] != '':
                        payload = {
                          "comment": {
                             "text": "{}".format(clean_comment(result[1])),
                          },
                          "languages": ["en"],
                          "requestedAttributes": {
                            "TOXICITY": {},
                            "SEVERE_TOXICITY": {}
                          }
                        }
                        payloads.append(payload)
                    else:
                        pass
                comment_ids = [comment[0] for comment in results]
                loop = asyncio.get_event_loop()
                future = asyncio.ensure_future(run(api_endpoint, payloads, headers, comment_ids))
                loop.run_until_complete(future)
                c.execute("SELECT unix from AskReddit WHERE toxicity IS NOT NULL ORDER BY unix DESC LIMIT 1;")
                last_utc = c.fetchall()[0][0]
                print(last_utc)
                print("Number of comments labelled: ", i*450)
                i += 1
            else:
                continue_label = False
            if (i % 4 == 0):
                conn.commit()
                conn.close()
                loop.close()
                continue_label = False
        except (KeyboardInterrupt, SystemExit):
            conn.commit()
            conn.close()
            loop.close()
            continue_label = False
        except Exception as e:
            print(str(e))
