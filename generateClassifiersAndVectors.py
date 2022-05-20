from sklearn import metrics, naive_bayes, svm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

import re
import string
import random

from nltk import word_tokenize
from nltk import FreqDist

import matplotlib.pyplot as plt

from collections import defaultdict

def cleanText(text):
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

    #remove emojis
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
                           "]+", flags = re.UNICODE)

    text = regrex_pattern.sub('', text)


    return text


def getStopWords():
    _stopwords = []
    with open("stop_words_english.txt", encoding="utf8") as fp:
        while True:
            
            line = fp.readline()
            _stopwords.append(line.strip())

            if "'" in line:
                _stopwords.append(line.replace("'", "").strip())
 
            if not line:
                break
        return _stopwords


def generateTokens(text):

    stopList =  getStopWords()
    
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if not t in stopList]

    return tokens

def generateCleanText(dataset):
    tokens = defaultdict(list)
    dataUpdated=[]
    cleanTexts = []
    
    
    for index, r in dataset.iterrows():
        
        label = r['category']
        #join headline short_description and full text
        text = r['headline'] + " " + r['short_description'] + " " + r['ftxt']
        text = text.lower()

        #idText = r[3]
        text = cleanText(text)
        text = generateTokens(text)

        doc_tokens = text
        txtConcac = ' '.join(doc_tokens)
        cleanTexts.append((label, txtConcac))
        
    return cleanTexts

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


def classifierEvaluation(classifier, vectorizer, xtest, ytest):
    xtext_tfidf = vectorizer.transform(xtest)
    ypred = classifier.predict(xtext_tfidf)

    average='macro'

    print('----macro----')
    precision = metrics.precision_score(ytest, ypred, average='macro')
    recall = metrics.recall_score(ytest, ypred, average='macro')
    f1 = metrics.f1_score(ytest, ypred, average='macro')
    accuracy = metrics.accuracy_score(ypred, ytest)
    print("precision ",precision)
    print("recall ",recall)
    print("f1 ",f1)
    print("accuracy ",accuracy)




def train_classifier(cleanTexts, classifierDumpName, vectDumpName):
    xtrain, xtest, ytrain, ytest = getSplits(cleanTexts)


    vectorizer = CountVectorizer(
    ngram_range=(1,3),
    min_df=3, analyzer='word')

    dtm = vectorizer.fit_transform(xtrain)

    naive_bayes_classifier = naive_bayes.MultinomialNB().fit(dtm, ytrain)

    classifierEvaluation(naive_bayes_classifier, vectorizer, xtest, ytest)

    xtext_tfidf = vectorizer.transform(xtest)
    ypred = naive_bayes_classifier.predict(xtext_tfidf)

    print("Naive Bayes Score -> ",accuracy_score(ypred, ytest)*100)


    file_naive_bayes = classifierDumpName
    pickle.dump(naive_bayes_classifier, open(file_naive_bayes, 'wb'))

    file_vectorizer = vectDumpName
    pickle.dump(vectorizer, open(file_vectorizer, 'wb'))
    
    cf_matrix = confusion_matrix(ypred, ytest, labels=naive_bayes_classifier.classes_)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=naive_bayes_classifier.classes_)
    
    return naive_bayes_classifier, vectorizer, disp

def generateCleanSentimentTxt(dataset):

    cleanTexts = []
    
    
    for index, r in dataset.iterrows():
        
        label = r['target']
        #join headline short_description and full text
        text = r['text']
        text = text.lower()

        #idText = r[3]
        text = cleanText(text)
        text = generateTokens(text)

        doc_tokens = text
        txtConcac = ' '.join(doc_tokens)
        cleanTexts.append((label, txtConcac))
        
    return cleanTexts

def generateCleanSentimentTxt(dataset):
    tokens = defaultdict(list)
    dataUpdated=[]
    cleanTexts = []
    
    
    for index, r in dataset.iterrows():
        
        label = r['target']
        #join headline short_description and full text
        text = r['text']
        text = text.lower()

        #idText = r[3]
        text = cleanText(text)
        text = generateTokens(text)

        doc_tokens = text
        txtConcac = ' '.join(doc_tokens)
        cleanTexts.append((label, txtConcac))
        
    return cleanTexts


txtDataset = pd.read_json('export_Joined_category_balanced.json')
cleanData = generateCleanText(txtDataset)
nb, vct, display = train_classifier(cleanData, 'naive_bayes_classifier.pkl', 'vectorizer.pkl')

sentimentDataset = pd.read_csv('training_sentiment.csv')
textSentimentClean = generateCleanSentimentTxt(sentimentDataset)
nb, vct, displaySentimentChart = train_classifier(textSentimentClean, 'naive_bayes_sentiment.pkl', 'sentiment_vectorizer.pkl')