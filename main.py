from asyncio.windows_events import NULL
import requests
from requests_oauthlib import OAuth1


from mysql.connector import connect, Error
cursor = NULL




url = 'https://api.twitter.com/1.1/statuses/home_timeline.json'
auth = OAuth1(' ', ' ', '85140814- ', ' ')
params = {'tweet_mode': 'extended',
'count': 200}
r =requests.get(url, params=params,auth=auth)
indx = 0

insert_tweets_query = """
INSERT INTO twitter_bubble.tweets
(text, username, language, timelineUserId, twitterUserId)
VALUES ( %s, %s, %s, %s, %s )
"""
tweetData = []

for tweet in r.json():
    indx = indx+1
    userinfo = tweet['user']
    print('---------   ' + str(indx) + '   ---------')
    print(userinfo['screen_name'])
    print (tweet['full_text'])
    print (tweet['lang'])
    print (tweet['created_at'])
    print(userinfo['id'])

    tweetData.append((tweet['full_text'], userinfo['screen_name'], tweet['lang'], 7, userinfo['id']))
    

try:
    with connect(
        host=" ",
        user=' ',
        password=' ',
    ) as connection:
        print(connection)

        with connection.cursor() as cursor:
            cursor.executemany(insert_tweets_query, tweetData)
            connection.commit()


except Error as e:
    print(e)

