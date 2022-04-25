import requests
from requests_oauthlib import OAuth1
import configparser
import sched, time
import threading
import databaseManipulator

class DataRequester(object):

    def __init__(self, interval=600):
        self.reqCount = 0

        self.interval = interval
        self._dataB = databaseManipulator.DatabaseManipulator()

        config = configparser.ConfigParser()
        # read the configuration file
        config.read('config.ini')
        self.__client_key = config.get('OAuth1', 'client_key')
        self.__client_secret = config.get('OAuth1', 'client_secret')
        self.__resource_owner_key = config.get('OAuth1', 'resource_owner_key')
        self.__resource_owner_secret = config.get('OAuth1', 'resource_owner_secret')

          
        thread = threading.Thread(target=self.requestTweets, args=())
        thread.daemon = True
        thread.start()
    
    def requestTweets(self):
        while True:
            print('Requesting Data!!! Request Number ' + str(self.reqCount))
            self.reqCount = self.reqCount + 1

            
            url = 'https://api.twitter.com/1.1/statuses/home_timeline.json'
            auth = OAuth1(self.__client_key, self.__client_secret,  self.__resource_owner_key, self.__resource_owner_secret)
            params = {'tweet_mode': 'extended',
            'count': 200}

            try:
                r =requests.get(url, params=params,auth=auth)
                r.raise_for_status()
                print(str(r.status_code))
                indx = 0
                tweetData = []
                for tweet in r.json():
                    userinfo = tweet['user']
                    #indx = indx+1
                    #userinfo = tweet['user']
                    #print('---------   ' + str(indx) + '   ---------')
                    #print(userinfo['screen_name'])
                    #print (tweet['full_text'])
                    #print (tweet['lang'])
                    #print (tweet['created_at'])
                    #print(userinfo['id'])
                    if tweet['lang'] == 'en':
                        tweetData.append((tweet['full_text'], userinfo['screen_name'], tweet['lang'], 7, userinfo['id'], tweet['id']))

                self._dataB.insertTweets(tweetData)

                print(str(len(tweetData)) + ' Tweets Loaded')


                #sc.enter(60, 1, self.requestTweets, (sc,))
            except requests.exceptions.HTTPError as err:
                raise SystemExit(err)

            time.sleep(self.interval)





    #def start(self):
    #    print('Start()')

     #   s = sched.scheduler(time.time, time.sleep)
      #  s.enter(60, 1, self.requestTweets, (s,))
       # s.run()




