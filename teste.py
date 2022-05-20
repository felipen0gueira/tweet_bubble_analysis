from cryptography.fernet import Fernet
import encrypDecrypService
from collections import defaultdict
import re
import string
from nltk import word_tokenize
import pandas as pd
from nltk import FreqDist
from sklearn import metrics, naive_bayes, svm
from sklearn.feature_extraction.text import CountVectorizer
import databaseManipulator


_stopwords = []
alldocs = []

import pickle





text ='Felipe N0gueira 😍😍😱😱👿👿🐔🌚  131313 09043-554 @felipen0g @flpNog https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python https://www.youtube.com/watch?v=7sz4WpkUIIs&t=335s 😍😍😱😱👿👿🐔🌚 '
print(text)
#remove numerals
patternNumeral = r'[0-9]'
text = re.sub(patternNumeral, '', text)

print(text)

patternUserName = r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)'
text = re.sub(patternUserName, '', text)

print(text)

patternURL = r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
text = re.sub(patternURL, '', text)

print(text)

regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
                           "]+", flags = re.UNICODE)

print('Removing: ' + regrex_pattern.sub('', text))



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





def get_tokens(text):

    stopList =  _stopwords #stopwords.words('english')
    
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if not t in stopList]

    return tokens
 


def clean_text(text):
    punct = string.punctuation+'“'+'’'+'‘'+'”'+'—'+'―'
    #text = text.replace('-', " ")
    text = text.translate(str.maketrans('', '', punct))

    pattern = r'[0-9]'

    text = re.sub(pattern, '', text)



    return text


def print_freq(myresult):
    tokens = defaultdict(list)
    dataUpdated=[]


    for i, row in myresult.iterrows():

        
        label = row['target']
        text = row['text']

        
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
        print(fd.most_common(100))


    

#loadStopWords()

#df = pd.read_csv('training.csv')

#print_freq(df)


objectRep = open("naive_bayes_classifier.pkl", "rb")
calssifier = pickle.load(objectRep)

objectSVM = open("SVM.pkl", "rb")
calssifierSVM = pickle.load(objectSVM)

vectorizerFile = open("vectorizer.pkl", "rb")
vectorizer = pickle.load(vectorizerFile)

result = calssifier.predict(vectorizer.transform(['Ex-Labor Secretary Says Second American Civil War Has Begun', 'Despite prices falling at the pump, annual Irish consumer price inflation hit 7% in April — a 22-year high',
'Fatality could have occurred when Luas collided with bridge, investigation finds The incident occurred on the approach to Beresford Place railway bridge in Dublin city last year.',
'An Cailín Ciúin: It shows an Irish language film can speak to people - it seems foolish to think it couldn We chat to director Colm Bairéad about making the film An Cailín Ciúin, based on a novella by Claire Keegan.']))

for i in result:
    print(i)


objectSent = open("Sent_naive_bayes_classifier.pkl", "rb")
calssifierSent = pickle.load(objectSent)

Sent_vectorizerFile = open("Sent_vectorizer.pkl", "rb")
sent_vectorizer = pickle.load(Sent_vectorizerFile)


result = calssifierSent.predict(sent_vectorizer.transform(['I am Glad', "I Hate my job", "Love Music","Ex-Labor Secretary Says Second American Civil War Has Begun",
"drink coca cola once a week. Because too much soda will be bad for me. He can drink juice or water. I will be the one drinking coke everyday. Because I'm the adult. lol. (2/2)"]))

for i in result:
    print(i)



