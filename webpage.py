from flask import Flask, render_template, redirect, session,request
import configparser
import requests

from requests_oauthlib import OAuth1Session
import databaseManipulator

app = Flask(__name__)

config = configparser.ConfigParser()
        # read the configuration file
config.read('config.ini')
client_key = config.get('OAuth1', 'client_key')
client_secret = config.get('OAuth1', 'client_secret')






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

@app.route("/logout", methods=['GET'])
def logout():
    session.pop("loggedUser")
    return redirect('/')


@app.route("/initAuth", methods=['GET'])
def initAuth():
    print('initAuth')
    url = 'https://api.twitter.com/oauth/request_token'
    params = {'oauth_callback': 'http://192.168.0.241:80'}
    auth = OAuth1Session(client_key, client_secret)

    #r=requests.post(url, params, auth)

    try:
        data = auth.get(url)
        print(data.text)
        splitedParams = str.split(data.text, '&')
        #r =requests.get(url, params=params,auth=auth)
        #print(str(r.status_code))
        
        #r.raise_for_status()
        

    except requests.exceptions.HTTPError as err:
        print('ERROR1')
        raise SystemExit(err)


    return redirect("https://api.twitter.com/oauth/authenticate?"+splitedParams[0])

@app.route("/getAccess", methods=['GET'])
def getAccessToken():
    print('/getAccess')
    url = 'https://api.twitter.com/oauth/access_token'

    params = {'oauth_token':request.args.get('oauth_token'),
    'oauth_verifier':request.args.get('oauth_verifier')
    }

    r =requests.get(url, params=params)

    print(request.args.get('oauth_token'))
    print(request.args.get('oauth_verifier'))

    r.raise_for_status()
    print(str(r.status_code))
    print(r.headers['content-type'])
    value =r.text.split('&')

    print(value)

    session['loggedUser'] = value[3].split('=')[1]

    db = databaseManipulator.DatabaseManipulator()

    db.insertUpdateUser((value[3].split('=')[1], value[0].split('=')[1], value[1].split('=')[1], value[2].split('=')[1]))

    

    return redirect("/main")

if(__name__=="__main__") :
    app.secret_key = "MY_KEY_APP"
    app.run(host='0.0.0.0', port=84)