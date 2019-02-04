import asyncio
from aiohttp import ClientSession
import sqlite3
import json

conn = sqlite3.connect("database/AskReddit_2008 - 2011.db")
c = conn.cursor()

headers = {}
api_token = "AIzaSyBma7dmCTo2Leiu56M5pWzhEA3CW_eu0Fk"
api_endpoint = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={}".format(api_token)

last_utc = 0
sql_transaction = []

continue_label = True

async def fetch(url, session, data, headers):
    async with session.post(url, data=data, headers=headers) as response:
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
    if len(sql_transaction) > 1999:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except Exception as e:
                print('SQL Transaction', str(e))
        conn.commit()
        sql_transaction = []

async def run(api_endpoint, payloads, headers, comment_ids):
    tasks = []
    async with ClientSession() as session:
        for payload in payloads:
            task = asyncio.ensure_future(fetch(api_endpoint, session, payload, headers))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        print("Number of Responses: ", len(responses))
        for response, comment_id in zip(responses, comment_ids):
            toxicity = json.loads(response)["attributeScores"]["TOXICITY"]['spanScores'][0]['score']['value']
            severe_toxic = json.loads(response)["attributeScores"]["SEVERE_TOXICITY"]['spanScores'][0]['score']['value']
            sql_update(toxicity, severe_toxic, comment_id)
i = 0
while continue_label == True:
    payloads = []
    c.execute('SELECT comment_id, comment, unix FROM {} WHERE unix > {} LIMIT 2000'.format("AskReddit", last_utc))
    results = c.fetchall()
    if len(results) > 0:
        for result in (comment for comment in results):
            payload = {
              "comment": {
                 "text": "{}".format(result[1]),
              },
              "languages": ["en"],
              "requestedAttributes": {
                "TOXICITY": {},
                "SEVERE_TOXICITY": {}
              }
            }
            payloads.append(json.dumps(payload))
        comment_ids = [comment[0] for comment in results]
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(run(api_endpoint, payloads, headers, comment_ids))
        loop.run_until_complete(future)
        last_utc = results[-1][2]
        print("Number of comments labelled: ", i*2000)
        i += 1
    elif i == 2:
        continue_label = False
    else:
        continue_label = False