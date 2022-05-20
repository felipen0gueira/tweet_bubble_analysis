import configparser
import pickle

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
            with connect(
                host=self.__host,
                user=self.__user,
                password=self.__password,
                database=self.__database,
                port=self.__port
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

    def insertUpdateUser(self, user):

        insert_user_query = """
        INSERT INTO user
        (username, accessToken, tokenSecret, twitterUserId)
        VALUES ( %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE username=%s, accessToken=%s, tokenSecret=%s
        """

        try:
            with connect(
                host=self.__host,
                user=self.__user,
                password=self.__password,
                database=self.__database,
                port=self.__port
            ) as connection:
                print('---connection status---')
                print(connection)

                with connection.cursor() as cursor:
                    cursor.execute(
                        insert_user_query, (user[0], user[1], user[2], user[3], user[0], user[1], user[2]))
                    connection.commit()

        except Error as e:
            print('---connection status---')
            print(e)

    def getUsersList(self):

        select_users = """
        SELECT * FROM twitter_bubble.user
        """

        try:
            with connect(
                host=self.__host,
                user=self.__user,
                password=self.__password,
                database=self.__database,
                port=self.__port
            ) as connection:
                print('---connection status---')
                print(connection)

                with connection.cursor() as cursor:
                    cursor.execute(select_users)
                    myresult = cursor.fetchall()
                    return myresult

        except Error as e:
            print('---connection status---')
            print(e)

    def getCategoriesClassification(self, userId):

        select_users = """
        select classification, count(*) from twitter_bubble.tweets where timelineUserId =%s and classification is not null group by classification;
        """

        try:
            with connect(
                host=self.__host,
                user=self.__user,
                password=self.__password,
                database=self.__database,
                port=self.__port
            ) as connection:
                print('---connection status---')
                print(connection)

                with connection.cursor() as cursor:
                    cursor.execute(select_users, [userId])
                    result = cursor.fetchall()
                    print('Results: ' + str(cursor.rowcount))

                    return result

        except Error as e:
            print('---connection status---')
            print(e)

    def getUserTweetPerClassification(self, userId):

        select_users = """
        select username, classification,COUNT(username) from twitter_bubble.tweets where timelineUserId =%s and classification is not null  group by username, classification
        """

        try:
            with connect(
                host=self.__host,
                user=self.__user,
                password=self.__password,
                database=self.__database,
                port=self.__port
            ) as connection:
                print('---connection status---')
                print(connection)

                with connection.cursor() as cursor:
                    cursor.execute(select_users, [userId])
                    result = cursor.fetchall()
                    print('Results: ' + str(cursor.rowcount))

                    return result

        except Error as e:
            print('---connection status---')
            print(e)

    def getUserByTwitterId(self, twitterId):

        select_users = """
        select * from twitter_bubble.user where twitterUserId =%s ;
        """

        try:
            with connect(
                host=self.__host,
                user=self.__user,
                password=self.__password,
                database=self.__database,
                port=self.__port
            ) as connection:
                print('---connection status---')
                print(connection)

                with connection.cursor() as cursor:
                    cursor.execute(select_users, [twitterId])
                    result = cursor.fetchall()
                    print('Results: ' + str(cursor.rowcount))
                    return result

        except Error as e:
            print('---connection status---')
            print(e)

    def getTweets(self, apUserID, category, userName=None):

        if(userName == None):
            select_tweets = """
            SELECT username, text, classification, sentiment FROM twitter_bubble.tweets where timelineUserId =%s and classification =%s order by createdAt desc
            """
        else:
            select_tweets = """
            SELECT username, text, classification, sentiment FROM twitter_bubble.tweets where timelineUserId =%s and classification =%s and username=%s order by createdAt desc
            """
        try:
            with connect(
                host=self.__host,
                user=self.__user,
                password=self.__password,
                database=self.__database,
                port=self.__port
            ) as connection:
                print('---connection status---')
                print(connection)

                whereData = [apUserID, category, userName] if userName != None else [
                    apUserID, category]

                with connection.cursor() as cursor:
                    cursor.execute(select_tweets, whereData)

                    result = cursor.fetchall()
                    print('Results: ' + str(cursor.rowcount))

                    return result

        except Error as e:
            print('---connection status---')

            print(e)

    def updateTextClassificationAndSentiment(self):

        naiveBTxtClassFile = open("naive_bayes_classifier.pkl", "rb")
        nbTxtClassifier = pickle.load(naiveBTxtClassFile)

        nbVectFile = open("vectorizer.pkl", "rb")
        nbVectorizer = pickle.load(nbVectFile)

        nbSntFile = open("naive_bayes_sentiment.pkl", "rb")
        nbSentClass = pickle.load(nbSntFile)

        nBVectFile = open("sentiment_vectorizer.pkl", "rb")
        sntVectFile = pickle.load(nBVectFile)

        try:
            with connect(
                host=self.__host,
                user=self.__user,
                password=self.__password,
                database=self.__database,
                port=self.__port
            ) as connection:
                print('---connection status---')
                print(connection)

                with connection.cursor() as cursor:
                    cursor.execute(
                        "select * from  twitter_bubble.tweets where classification is null or sentiment is null")
                    myresult = cursor.fetchall()

                    textVct = []
                    idsVect = []
                    for i in myresult:
                        textVct.append(i[1])
                        idsVect.append(i[0])

                    updatedTweets = []
                    predid = nbTxtClassifier.predict(
                        nbVectorizer.transform(textVct))
                    sentimentPredicted = nbSentClass.predict(
                        sntVectFile.transform(textVct))
                    for p in range(len(textVct)):
                        print(idsVect[p] + ' - ' + textVct[p] + ' - ' +
                              predid[p] + ' - Sentiment ' + str(sentimentPredicted[p]))
                        print(
                            '\n\n-----------------------------------------------------------------')
                        updatedTweets.append(
                            [predid[p], sentimentPredicted[p].item(), idsVect[p]])

                    self.updateTweet(updatedTweets)

        except Error as e:
            print('---connection status---')
            print(e)



    def updateTweet(self, dataUpdated):

        updateRows = """
        UPDATE twitter_bubble.tweets SET classification = %s, sentiment = %s WHERE idtweets like %s
        """

        try:
            with connect(
                    host=self.__host,
                    user=self.__user,
                    password=self.__password,
                    database=self.__database,
                    port=self.__port
                ) as connection:
                    print('---updating---')
                    print(connection)

                    with connection.cursor() as cursor:
                        cursor.executemany(updateRows, dataUpdated)
                        print('Tweets updated : ' + str(cursor.rowcount))
                        connection.commit()
                        print(cursor.statement)
                        print('Tweets updated : ' + str(cursor.rowcount))

        except Error as e:
            print('---connection status---')
            print(e)
