from asyncio.windows_events import NULL
from collections import defaultdict
from email.policy import default
import json
import configparser

from numpy import vectorize
from sklearn import metrics, naive_bayes, svm
from sklearn.metrics import accuracy_score

import nltk
import string
import random
import re


from sklearn.feature_extraction.text import CountVectorizer



from mysql.connector import connect, Error
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import FreqDist
alldocs = []

_naive_bayes_classifier = NULL
_vectorizer = NULL
_stopwords = []


def loadStopWords():
    with open("stop_words_english.txt", encoding="utf8") as fp:
        while True:
            
            line = fp.readline()
            _stopwords.append(line.strip())

            if "'" in line:
                _stopwords.append(line.replace("'", "").strip())
 
            if not line:
                break

        
        _stopwords.append('huffpost')
        _stopwords.append('facebook')
        _stopwords.append('twitter')
        _stopwords.append('times')
        _stopwords.append('time')
        _stopwords.append('huffpoststyle')
        _stopwords.append('people')
        _stopwords.append('year')
        _stopwords.append('years')
        _stopwords.append('day')
        _stopwords.append('days')
        
        for w in _stopwords:
            print(w)


def getSplits(docs):
    random.shuffle(docs)
    
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    pivot = int(.80 * len (docs))

    for i in range(0, pivot):
        x_train.append(docs[i][1])
        y_train.append(docs[i][0])


    for i in range(pivot, len(docs)):
        x_test.append(docs[i][1])
        y_test.append(docs[i][0])


    return x_train, x_test, y_train, y_test



def clean_text(text):
    punct = string.punctuation+'“'+'’'+'‘'+'”'+'—'+'―'
    #text = text.replace('-', " ")
    text = text.translate(str.maketrans('', '', punct))

    pattern = r'[0-9]'

    text = re.sub(pattern, '', text)



    return text

def get_tokens(text):

    stopList =  _stopwords #stopwords.words('english')
    
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if not t in stopList]

    return tokens
    
def print_freq(myresult):
    tokens = defaultdict(list)
    dataUpdated=[]


    for r in myresult:
        
        label = r[0]
        text = r[1] + " " + r[2] + " " + r[3]
        text = text.lower()
        #alldocs.append((label, text))
        idText = r[3]
        text = clean_text(text)
        text = get_tokens(text)
        #doc_tokens = word_tokenize(text)
        doc_tokens = text
        txtConcac = ' '.join(doc_tokens)
        alldocs.append((label, txtConcac))
        #dataUpdated.append((txtConcac, idText))
        #print('ID: ' + str(idText))
        #print(' '.join(map(str, doc_tokens))) 
        #print(doc_tokens)
 
        tokens[label].extend(doc_tokens)
    
    #xtrain, xtest, ytrain, ytest = getSplits(alldocs)

    print(tokens['crime'])




    

    
    config = configparser.ConfigParser()
    # read the configuration file
    #config.read('config.ini')

    #host = config.get('DB', 'host')
    #user = config.get('DB', 'user')
    #password = config.get('DB', 'password')
    #database = config.get('DB', 'database')
    #port = config.get('DB', 'port')


    #updateRows = """
    #UPDATE twitter_bubble.training_dataset SET filteredText = %s WHERE idtraining_dataset = %s
    #"""

            
    #try:
    #    with connect(
    #                host = host,
    #                user = user,
    #                password = password,
    #                database = database,
    #                port = port
    #            ) as connection:
    #                print('---updating---')
    #                print(connection)

    #                with connection.cursor() as cursor:
    #                    cursor.executemany(updateRows, dataUpdated)
    #                    connection.commit()


    #except Error as e:
    #        print('---connection status---')
    #        print(e)



    

    for category_label, category_tokens in tokens.items():
        print (category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(1000))


    





def loadData():
    config = configparser.ConfigParser()
    # read the configuration file
    config.read('config.ini')

    host = config.get('DB', 'host')
    user = config.get('DB', 'user')
    password = config.get('DB', 'password')
    database = config.get('DB', 'database')
    port = config.get('DB', 'port')

    count = 0 
    dataS = []


        
    try:
        with connect(
                    host = host,
                    user = user,
                    password = password,
                    database = database,
                    port = port
                ) as connection:
                    print('---connection status---')
                    print(connection)

                    with connection.cursor() as cursor:
                        cursor.execute("select category, headline, short_description, ftxt, idtraining_dataset from (SELECT category, headline, short_description, ftxt, idtraining_dataset, filteredText, CHAR_LENGTH(filteredText) as CharLen, row_number() over (partition by category order by CHAR_LENGTH(filteredText) desc) as numbRow  FROM twitter_bubble.training_dataset where ftxt is not null and ftxt not like '') ranked where numbRow <= 1000")
                        #cursor.execute("SELECT category, headline, short_description, ftxt, idtraining_dataset FROM twitter_bubble.training_dataset where ftxt is not null or ftxt not like ''")
                        #cursor.execute("SELECT category, filteredText FROM twitter_bubble.training_dataset")
                        myresult = cursor.fetchall()
                        #print_freq(myresult)
                        return myresult


    except Error as e:
            print('---connection status---')
            print(e)



def evaluate_classifier(title, classifier, vectorizer, xtest, ytest):
    xtext_tfidf = vectorizer.transform(xtest)
    ypred = classifier.predict(xtext_tfidf)
    
    precision = metrics.precision_score(ytest, ypred, average='micro')
    recall = metrics.recall_score(ytest, ypred, average='micro')
    f1 = metrics.f1_score(ytest, ypred, average='micro')

    print("%s\t%f\t%f\t%f\n" % (title, precision,recall, f1))



def train_classifier(docs):
    xtrain, xtest, ytrain, ytest = getSplits(alldocs)

    vectorizer = CountVectorizer(stop_words='english',
    ngram_range=(1,3),
    min_df=3, analyzer='word')

    dtm = vectorizer.fit_transform(xtrain)

    naive_bayes_classifier = naive_bayes.MultinomialNB().fit(dtm, ytrain)

    evaluate_classifier("Naive Bayes TRAIN", naive_bayes_classifier, vectorizer, xtrain,ytrain)
    evaluate_classifier("Naive Bayes TEST", naive_bayes_classifier, vectorizer, xtest, ytest)

    #print("Naive Bayes Accuracy Score -> ",accuracy_score(naive_bayes_classifier, ytest)*100)

    #textVect = ['the new born is so cute', 'new single']
    #predid = naive_bayes_classifier.predict(vectorizer.transform(textVect))
    #I don't work under pressure
    _naive_bayes_classifier = naive_bayes_classifier
    _vectorizer = vectorizer



    print('Classifier - Algorithm - SVM')
    print('fit the training dataset on the classifier')
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(dtm, ytrain)
    print('predict the labels on validation dataset')
    xtext_tfidf = vectorizer.transform(xtest)
    predictions_SVM = SVM.predict(xtext_tfidf)
    print('Use accuracy_score function to get the accuracy') 
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, ytest)*100)


    #classifyTweets(naive_bayes_classifier, vectorizer)

    #for p in predid:
    #    print(p)

    #while True:
    #    value = input("Please enter a text:\n")
    #    predid = naive_bayes_classifier.predict(vectorizer.transform([value]))
    #    print(predid[0])

def updateTweet(dataUpdated):
    config = configparser.ConfigParser()
    # read the configuration file
    config.read('config.ini')

    host = config.get('DB', 'host')
    user = config.get('DB', 'user')
    password = config.get('DB', 'password')
    database = config.get('DB', 'database')
    port = config.get('DB', 'port')


    updateRows = """
    UPDATE twitter_bubble.tweets SET classification = %s WHERE idtweets = %s
    """

            
    try:
        with connect(
                    host = host,
                    user = user,
                    password = password,
                    database = database,
                    port = port
                ) as connection:
                    print('---updating---')
                    print(connection)

                    with connection.cursor() as cursor:
                        cursor.execute(updateRows, dataUpdated)
                        connection.commit()


    except Error as e:
            print('---connection status---')
            print(e)


def classifyTweets(naive_bayes_classifier, vectorizer):

        config = configparser.ConfigParser()
        # read the configuration file
        config.read('config.ini')

        host = config.get('DB', 'host')
        user = config.get('DB', 'user')
        password = config.get('DB', 'password')
        database = config.get('DB', 'database')
        port = config.get('DB', 'port')

        count = 0 
        dataS = []


            
        try:
            with connect(
                        host = host,
                        user = user,
                        password = password,
                        database = database,
                        port = port
                    ) as connection:
                        print('---connection status---')
                        print(connection)

                        with connection.cursor() as cursor:
                            cursor.execute("select idtweets, text, classification from tweets WHERE language = 'en' ")
                            #cursor.execute("SELECT category, headline, short_description, idtraining_dataset FROM twitter_bubble.training_dataset ")
                            #cursor.execute("SELECT category, filteredText FROM twitter_bubble.training_dataset")
                            myresult = cursor.fetchall()
                            #print_freq(myresult)
                            textVct = []
                            idsVect = []
                            for i in myresult:
                                textVct.append(i[1])
                                idsVect.append(i[0])


                            predid = naive_bayes_classifier.predict(vectorizer.transform(textVct))
                            for p in range(len(textVct)):
                                print(idsVect[p] + ' - ' + textVct[p] + ' - ' + predid[p])
                                print('\n\n-----------------------------------------------------------------')
                                updateTweet((predid[p], idsVect[p]))


                            return myresult


        except Error as e:
                print('---connection status---')
                print(e)





if __name__ == '__main__':
    loadStopWords()
    dbData = loadData()
    print('Number of rows: ' + str(len(dbData)))
    print_freq(dbData)

    train_classifier(alldocs)

    



#with open("Category_Dataset.json") as fp:
#    while True:
#        count += 1
#        line = fp.readline()
# 
#        if not line:
#            break

#        json_object = json.loads(line.strip())
#        dataS.append((json_object["category"], json_object["headline"],json_object["authors"], json_object["link"],json_object["short_description"],json_object["date"]))
#        print(str(count))

#    insert_tweets_query = """
#        INSERT IGNORE INTO training_dataset
#        (category, headline, authors, link, short_description, date)
#        VALUES ( %s, %s, %s, %s, %s, %s)
#        """

    
#    try:
#        with connect(
#                host = host,
#                user = user,
#                password = password,
#                database = database,
#                port = port
#            ) as connection:
#                print('---connection status---')
#                print(connection)

#                with connection.cursor() as cursor:
#                    cursor.executemany(insert_tweets_query, dataS)
#                    connection.commit()
#
#
#    except Error as e:
#        print('---connection status---')
#        print(e)

