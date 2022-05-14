from asyncio.windows_events import NULL
from collections import defaultdict
from email.policy import default
import pandas as pd
import configparser

from numpy import vectorize
from sklearn import metrics, naive_bayes, svm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle

import nltk
import string
import random
import re
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer

from sklearn import preprocessing

from nltk.stem import SnowballStemmer



from mysql.connector import connect, Error
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import FreqDist


from sklearn.neural_network import MLPClassifier


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

        
        #_stopwords.append('huffpost')
        #_stopwords.append('facebook')
        #_stopwords.append('twitter')
        #_stopwords.append('times')
        #_stopwords.append('time')
        #_stopwords.append('huffpoststyle')
        #_stopwords.append('people')
        #_stopwords.append('year')
        #_stopwords.append('years')
        #_stopwords.append('day')
        #_stopwords.append('days')
        


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
    text = text.translate(str.maketrans('', '', punct))

    #remove numerals
    pattern = r'[0-9]'
    text = re.sub(pattern, '', text)

    #remove usernames
    patternUserName = r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)'
    text = re.sub(patternUserName, '', text)

    #remove URL
    patternURL = r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
    text = re.sub(patternURL, '', text)

    
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
                           "]+", flags = re.UNICODE)

    text = regrex_pattern.sub('', text)


    return text

def get_tokens(text):

    stopList =  _stopwords #stopwords.words('english')
    
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if not t in stopList]

    return tokens


def executeStemmer(wordList):
    snowball = SnowballStemmer(language='english')
    steemedWords = []

    for word in wordList:
        steemedWords.append(snowball.stem(word))

    return steemedWords

    
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
        #text = executeStemmer(text)
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






    

    
    config = configparser.ConfigParser()
  
    

    for category_label, category_tokens in tokens.items():
        print (category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(10))


    





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
                        cursor.execute("select category, headline, short_description, ftxt, idtraining_dataset from (SELECT category, headline, short_description, ftxt, idtraining_dataset, filteredText, CHAR_LENGTH(filteredText) as CharLen, row_number() over (partition by category order by CHAR_LENGTH(filteredText) desc) as numbRow  FROM twitter_bubble.training_dataset where ftxt is not null and ftxt not like '') ranked where numbRow <= 1100")
                        #cursor.execute("SELECT category, headline, short_description, ftxt, idtraining_dataset FROM twitter_bubble.training_dataset_2 where ftxt is not null or ftxt not like ''")
                        #cursor.execute("SELECT category, headline, short_description, ftxt, idtraining_dataset FROM twitter_bubble.training_dataset_2 where ftxt is not null and ftxt not like ''")
                        myresult = cursor.fetchall()
                        #print_freq(myresult)
                        return myresult


    except Error as e:
            print('---connection status---')
            print(e)



def evaluate_classifier(title, classifier, vectorizer, xtest, ytest):
    xtext_tfidf = vectorizer.transform(xtest)
    ypred = classifier.predict(xtext_tfidf)
    print('----micro----')
    precision = metrics.precision_score(ytest, ypred, average='micro')
    recall = metrics.recall_score(ytest, ypred, average='micro')
    f1 = metrics.f1_score(ytest, ypred, average='micro')
    print("%s\t%f\t%f\t%f\n" % (title, precision,recall, f1))

    print('----macro----')
    precision = metrics.precision_score(ytest, ypred, average='macro')
    recall = metrics.recall_score(ytest, ypred, average='macro')
    f1 = metrics.f1_score(ytest, ypred, average='macro')
    print("%s\t%f\t%f\t%f\n" % (title, precision,recall, f1))
    print('----weighted---')
    precision = metrics.precision_score(ytest, ypred, average='weighted')
    recall = metrics.recall_score(ytest, ypred, average='weighted')
    f1 = metrics.f1_score(ytest, ypred, average='weighted')
    print("%s\t%f\t%f\t%f\n" % (title, precision,recall, f1))





def train_classifier(docs, file_naive_bayes = 'naive_bayes_classifier.pkl', file_vectorizer = 'vectorizer.pkl'):
    xtrain, xtest, ytrain, ytest = getSplits(alldocs)
    print('xtrain ' + str(len(xtrain)))
    print('xtest ' + str(len(xtest)))
    print('ytrain ' + str(len(ytrain)))
    print('ytest ' + str(len(ytest)))

 
#_stopwords
    vectorizer = CountVectorizer(stop_words=_stopwords,
    ngram_range=(1,3),
    min_df=3, analyzer='word')

    dtm = vectorizer.fit_transform(xtrain)

    naive_bayes_classifier = naive_bayes.MultinomialNB().fit(dtm, ytrain)

    evaluate_classifier("Naive Bayes TRAIN", naive_bayes_classifier, vectorizer, xtrain,ytrain)
    evaluate_classifier("Naive Bayes TEST", naive_bayes_classifier, vectorizer, xtest, ytest)

    xtext_tfidf = vectorizer.transform(xtest)
    ypred = naive_bayes_classifier.predict(xtext_tfidf)

    print("Naive Bayes Score -> ",accuracy_score(ypred, ytest)*100)

    #print("Naive Bayes Accuracy Score -> ",accuracy_score(naive_bayes_classifier, ytest)*100)

    #textVect = ['the new born is so cute', 'new single']
    #predid = naive_bayes_classifier.predict(vectorizer.transform(textVect))
    #I don't work under pressure
    _naive_bayes_classifier = naive_bayes_classifier
    _vectorizer = vectorizer

    #file_naive_bayes = 'naive_bayes_classifier.pkl'
    pickle.dump(naive_bayes_classifier, open(file_naive_bayes, 'wb'))

    #file_vectorizer = 'vectorizer.pkl'
    pickle.dump(vectorizer, open(file_vectorizer, 'wb'))





def train_classifierSVM(docs):
    xtrain, xtest, ytrain, ytest = getSplits(alldocs)
    print('xtrain ' + str(len(xtrain)))
    print('xtest ' + str(len(xtest)))
    print('ytrain ' + str(len(ytrain)))
    print('ytest ' + str(len(ytest)))

 
#_stopwords
    vectorizer = CountVectorizer(stop_words=_stopwords,
    ngram_range=(1,3),
    min_df=3, analyzer='word')


 


    print('Classifier - Algorithm - SVM')
    print('fit the training dataset on the classifier')
    dtm = vectorizer.fit_transform(xtrain)
    SVM = LinearSVC()#svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    svmclasssifier=SVM.fit(dtm, ytrain)
    print('predict the labels on validation dataset')
    xtext_tfidf = vectorizer.transform(xtest)
    predictions_SVM = SVM.predict(xtext_tfidf)
    print('Use accuracy_score function to get the accuracy') 
    for p, r, txt in zip(predictions_SVM, ytest, xtest):
        print(p, ',', r, ',', str(p==r) , txt)
    
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, ytest)*100)


    file_vectorizer = 'vectorizer.pkl'
    pickle.dump(vectorizer, open(file_vectorizer, 'wb'))
    
    file_SVM = 'SVM.pkl'
    pickle.dump(SVM, open(file_SVM, 'wb'))


    result = SVM.predict(vectorizer.transform(['Ex-Labor Secretary Says Second American Civil War Has Begun', 'Despite prices falling at the pump, annual Irish consumer price inflation hit 7% in April — a 22-year high',
'Fatality could have occurred when Luas collided with bridge, investigation finds The incident occurred on the approach to Beresford Place railway bridge in Dublin city last year.',
'An Cailín Ciúin: It shows an Irish language film can speak to people - it seems foolish to think it couldn We chat to director Colm Bairéad about making the film An Cailín Ciúin, based on a novella by Claire Keegan.']))

    for i in result:
        print(i)







    print('----micro SVM----')
    precision = metrics.precision_score(ytest, predictions_SVM, average='micro')
    recall = metrics.recall_score(ytest, predictions_SVM, average='micro')
    f1 = metrics.f1_score(ytest, predictions_SVM, average='micro')
    print("%s\t%f\t%f\t%f\n" % ('SVM', precision,recall, f1))
    
    print('----macro SVM----')
    precision = metrics.precision_score(ytest, predictions_SVM, average='macro')
    recall = metrics.recall_score(ytest, predictions_SVM, average='macro')
    f1 = metrics.f1_score(ytest, predictions_SVM, average='macro')
    print("%s\t%f\t%f\t%f\n" % ('SVM', precision,recall, f1))
    print('----weighted SVM---')
    precision = metrics.precision_score(ytest, predictions_SVM, average='weighted')
    recall = metrics.recall_score(ytest, predictions_SVM, average='weighted')
    f1 = metrics.f1_score(ytest, predictions_SVM, average='weighted')
    print("%s\t%f\t%f\t%f\n" % ('SVM', precision,recall, f1))










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
    UPDATE twitter_bubble.tweets SET classification = %s WHERE idtweets like %s
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
                        cursor.executemany(updateRows, dataUpdated)
                        print('Tweets updated : ' + str(cursor.rowcount))
                        connection.commit()
                        print(cursor.statement)
                        print('Tweets updated : ' + str(cursor.rowcount))


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
                            cursor.execute("select * from  twitter_bubble.tweets WHERE classification is null")
                            #cursor.execute("SELECT category, headline, short_description, idtraining_dataset FROM twitter_bubble.training_dataset ")
                            #cursor.execute("SELECT category, filteredText FROM twitter_bubble.training_dataset")
                            myresult = cursor.fetchall()
                            #print_freq(myresult)
                            textVct = []
                            idsVect = []
                            for i in myresult:
                                textVct.append(i[1])
                                idsVect.append(i[0])

                            updatedTweets = []
                            predid = naive_bayes_classifier.predict(vectorizer.transform(textVct))
                            for p in range(len(textVct)):
                                print(idsVect[p] + ' - ' + textVct[p] + ' - ' + predid[p])
                                print('\n\n-----------------------------------------------------------------')
                                updatedTweets.append([predid[p], idsVect[p]])


                            updateTweet(updatedTweets)


                            return myresult


        except Error as e:
                print('---connection status---')
                print(e)



def print_frequenci(myresult):
    tokens = defaultdict(list)
    dataUpdated=[]

    for index, r in myresult.iterrows():
        
        label = r['target']
        text = r['text']
        text = text.lower()
       
        text = clean_text(text)
        text = get_tokens(text)
        doc_tokens = text
        txtConcac = ' '.join(doc_tokens)
        alldocs.append((label, txtConcac))

 
        tokens[label].extend(doc_tokens)
        

    for category_label, category_tokens in tokens.items():
        print (category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(50))



def generateSentimentClassifier():
    print('Generating Sentiment Classifier')
    sentimentDataset = pd.read_csv('training_sentiment.csv')
    negative = sum(sentimentDataset['target'] == 0)

    print('Negative = ' + str(negative))

    positive = sum(sentimentDataset['target'] == 4)
    print('Positive = ' + str(positive))

    print_frequenci(sentimentDataset)

def generateFilteredText():
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
                        cursor.execute("select category, headline, short_description, ftxt, idtraining_dataset from (SELECT category, headline, short_description, ftxt, idtraining_dataset, filteredText, CHAR_LENGTH(filteredText) as CharLen, row_number() over (partition by category order by CHAR_LENGTH(filteredText) desc) as numbRow  FROM twitter_bubble.training_dataset where ftxt is not null and ftxt not like '') ranked where numbRow <= 1100")
                        #cursor.execute("SELECT category, headline, short_description, ftxt, idtraining_dataset FROM twitter_bubble.training_dataset_2 where ftxt is not null or ftxt not like ''")
                        #cursor.execute("SELECT category, headline, short_description, ftxt, idtraining_dataset FROM twitter_bubble.training_dataset_2 where ftxt is not null and ftxt not like ''")
                        myresult = cursor.fetchall()
                        #print_freq(myresult)
                        
                        for r in myresult:
                            
                            label = r[0]
                            text = r[1] + " " + r[2] + " " + r[3]
                            text = text.lower()
                            #alldocs.append((label, text))
                            idText = r[3]
                            text = clean_text(text)
                            text = get_tokens(text)
                            #text = executeStemmer(text)
                            #doc_tokens = word_tokenize(text)
                            doc_tokens = text
                            txtConcac = ' '.join(doc_tokens)
                            alldocs.append((label, txtConcac))
                            #print('---------------------------------------------------------------------------------------------')
                            #print(txtConcac)
                            #dataUpdated.append((txtConcac, idText))
                            #print('ID: ' + str(idText))
                            #print(' '.join(map(str, doc_tokens))) 
                            #print(doc_tokens)
                        return myresult


    except Error as e:
            print('---connection status---')
            print(e)





if __name__ == '__main__':
    #loadStopWords()
    #generateSentimentClassifier()
    #dbData = loadData()
    #print('Number of rows: ' + str(len(dbData)))
    #print_freq(dbData)
    #train_classifier(alldocs)

    objectRep = open("naive_bayes_classifier.pkl", "rb")
    calssifier = pickle.load(objectRep)

    vectorizerFile = open("vectorizer.pkl", "rb")
    vectorizer = pickle.load(vectorizerFile)

    classifyTweets(calssifier, vectorizer)

    #train_classifier(alldocs, file_naive_bayes = 'Sent_naive_bayes_classifier.pkl', file_vectorizer = 'Sent_vectorizer.pkl')
    #train_classifierSVM(alldocs)

    #generateFilteredText()
