from unittest import result
from flask import Flask, render_template, redirect, jsonify, session, request
import configparser
import requests

from requests_oauthlib import OAuth1Session
import databaseManipulator
import encrypDecrypService

app = Flask(__name__)

config = configparser.ConfigParser()
# read the configuration file
config.read('config.ini')
client_key = config.get('OAuth1', 'client_key')
client_secret = config.get('OAuth1', 'client_secret')


def validateUser():
    if 'userTwitterId' in session:
        return True
    else:
        return False


@app.route("/", methods=['GET'])
def mainPage():
    if session.get('loggedUser'):
        print(session.get('loggedUser'))
        return redirect("/main")

    else:
        return render_template("login.html")


@app.route("/main", methods=['GET'])
def main():
    if not session['loggedUser']:
        return redirect('/')

    data = {'user': '@'+session['loggedUser']}
    return render_template("main.html", data=data)



@app.route("/TweetsView", methods=['GET'])
def tweetsView():
    if not session['loggedUser']:
        return redirect('/')

    data = {'user': '@'+session['loggedUser']}
    return render_template("TweetsView.html", data=data)


@app.route("/logout", methods=['GET'])
def logout():
    session.pop("loggedUser")
    return redirect('/')


@app.route("/initAuth", methods=['GET'])
def initAuth():
    print('initAuth')
    url = 'https://api.twitter.com/oauth/request_token'

    auth = OAuth1Session(client_key, client_secret)

    try:
        data = auth.get(url)
        print(data.text)
        splitedParams = str.split(data.text, '&')


    except requests.exceptions.HTTPError as err:
        print('ERROR1')
        raise SystemExit(err)

    return redirect("https://api.twitter.com/oauth/authenticate?"+splitedParams[0])

@app.route("/loadTweets", methods=['GET'])
def loadTweets():
    if validateUser():

        category = request.args.get('category')
        user = request.args.get('user')

        print(category)
        print(user)
        if(category ==None):
            return jsonify({'invalid request'})

        if(user != None):
            user = user.replace('@','')


        db = databaseManipulator.DatabaseManipulator()

        loadedData = db.getTweets(session['appId'], category, user)
        data = {'user': '@'+session['loggedUser']}
        data["tweetsList"] = []

        titleLabel = str(len(loadedData))+" Tweets by @" + user + " in " + category if user != None else str(len(loadedData))+" in Category " + category

        data["ViewLabel"] = titleLabel

        for row in loadedData:
            tweet={
                'userName':'@'+row[0],
                'text':row[1],
                'category': row[2],
                'sentiment': row[3]
            }
            data["tweetsList"].append(tweet)
            


        return render_template("TweetsView.html", data=data)




@app.route("/getAccess", methods=['GET'])
def getAccessToken():
    print('/getAccess')
    url = 'https://api.twitter.com/oauth/access_token'

    params = {'oauth_token': request.args.get('oauth_token'),
              'oauth_verifier': request.args.get('oauth_verifier')
              }

    r = requests.get(url, params=params)

    r.raise_for_status()
    print(str(r.status_code))
    print(r.headers['content-type'])
    value = r.text.split('&')

    print(value)

    session['loggedUser'] = value[3].split('=')[1]
    session['userTwitterId'] = value[2].split('=')[1]

    db = databaseManipulator.DatabaseManipulator()

    db.insertUpdateUser((value[3].split('=')[1], encrypDecrypService.EncryptDecryptService.encrypt(value[0].split('=')[1]),
                         encrypDecrypService.EncryptDecryptService.encrypt(value[1].split('=')[1]), value[2].split('=')[1]))

    appUserId = db.getUserByTwitterId(session['userTwitterId'])
    session['appId'] = appUserId[0][0]

    return redirect("/main")


@app.route('/api/get-filter-bubble', methods=['GET'])
def getFilterBubble():
    if validateUser():
        db = databaseManipulator.DatabaseManipulator()
        id = session['appId']
        resultDb = db.getCategoriesClassification(id)

        fBublleData = {'name': "@"+session['loggedUser']+"'s Filter Bubbler",
                       "color": '#98A0A0'}

        children=[]

        classUserschildren = {}

        userTweetsPerClass = db.getUserTweetPerClassification(session['appId'])

        for u in userTweetsPerClass:
            if u[1] in classUserschildren:
                    classUserschildren[u[1]].append({        
                        "name": "@"+u[0],
                        "color": '#EBF3F4',
                        "size": u[2],
                        "type": 'user',
                        "category":u[1]
                        }) 
            else:
                classUserschildren[u[1]] = []
                classUserschildren[u[1]].append({        
                        "name": "@"+u[0],
                        "color": '#EBF3F4',
                        "size": u[2],
                        "type": 'user',
                        "category": u[1]}) 




            #classUserschildren[]

        for r in resultDb:
            children.append({
                "color": "#1181C8",
                "name":r[0],
                "size":r[1],
                "type":"category",
                "children":classUserschildren[r[0]]
            })
        
        fBublleData["children"] = children


        return jsonify(fBublleData)


if(__name__ == "__main__"):
    app.secret_key = "MY_KEY_APP"
    app.run(host='0.0.0.0', port=84)