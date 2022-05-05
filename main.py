
import requests
from requests_oauthlib import OAuth1
import configparser
import databaseManipulator
import dataRequester
import threading
import time

from mysql.connector import connect, Error



config = configparser.ConfigParser()

# read the configuration file
config.read('config.ini')

# load config vars
# DB
host = config.get('DB', 'host')
user = config.get('DB', 'user')
password = config.get('DB', 'password')
database = config.get('DB', 'database')


#OAUth
client_key = config.get('OAuth1', 'client_key')
client_secret = config.get('OAuth1', 'client_secret')
resource_owner_key = config.get('OAuth1', 'resource_owner_key')
resource_owner_secret = config.get('OAuth1', 'resource_owner_secret')


url = 'https://api.twitter.com/1.1/statuses/home_timeline.json'
auth = OAuth1(client_key, client_secret, resource_owner_key, resource_owner_secret)
params = {'tweet_mode': 'extended',
'count': 200}

#try:
    #r =requests.get(url, params=params,auth=auth)
    #r.raise_for_status()
#except requests.exceptions.HTTPError as err:
    #raise SystemExit(err)

#indx = 0

#print(str(r.status_code))

#dataB = databaseManipulator.DatabaseManipulator()
print('Init DataRequester')
dataReq = dataRequester.DataRequester()

print('starting dataReq!')
#threadFunc = threading.Thread(target=dataReq.start(), name="str")
#threadFunc.daemon = True
#threadFunc.start()

print('dataReq started!')

x = False
while not x:
     time.sleep(0.1)



#tweetData = []

#for tweet in r.json():
    #indx = indx+1
    #userinfo = tweet['user']
    #print('---------   ' + str(indx) + '   ---------')
    #print(userinfo['screen_name'])
    #print (tweet['full_text'])
    #print (tweet['lang'])
    #print (tweet['created_at'])
    #print(userinfo['id'])

#    tweetData.append((tweet['full_text'], userinfo['screen_name'], tweet['lang'], 7, userinfo['id'], tweet['id']))

#dataB.insertTweets(tweetData)