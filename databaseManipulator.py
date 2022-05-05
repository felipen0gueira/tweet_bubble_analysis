import configparser

from mysql.connector import connect, Error

class DatabaseManipulator:
    __host = ''
    __user = ''
    __password = ''
    __database = ''
    __port = ''

    def __init__(self):

        config = configparser.ConfigParser()
        # read the configuration file
        config.read('config.ini')

        # load config vars
        # DB
        self.__host = config.get('DB', 'host')
        self.__user = config.get('DB', 'user')
        self.__password = config.get('DB', 'password')
        self.__database = config.get('DB', 'database')
        self.__port = config.get('DB', 'port')
    
    def insertTweets(self, tweets):

        insert_tweets_query = """
        INSERT IGNORE INTO tweets
        (text, username, language, timelineUserId, twitterUserId, idtweets)
        VALUES ( %s, %s, %s, %s, %s, %s)
        """

        try:
            with connect( host = self.__host,
                user = self.__user,
                password = self.__password,
                database = self.__database,
                port = self.__port
            ) as connection:
                print('---connection status---')
                print(connection)

                with connection.cursor() as cursor:
                    cursor.executemany(insert_tweets_query, tweets)
                    connection.commit()
                    print('Tweets inserted : ' + str(cursor.rowcount))


        except Error as e:
            print('---connection status---')
            print(e)




