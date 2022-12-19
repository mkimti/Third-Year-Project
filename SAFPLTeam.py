import snscrape.modules.twitter as sntwitter
import pandas as pd
import sqlite3
from sqlite3 import Error
import ssl
import nltk
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import contractions
import math
from unidecode import unidecode
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pattern.en import sentiment
import os
from textblob import TextBlob
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

PATH = 'Fantasy-Premier-League/data'

def createPlayersSumInitialTeam(conn):
    c = conn.cursor()
    
    c.execute("""CREATE TABLE playersSumInitialTeam (
        pID INTEGER,
        ictIndexSum INTEGER,
        FOREIGN KEY (pID) REFERENCES players(playerId)
    )""")

    conn.commit()

def createPlayersTable(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE players (
        playerId INTEGER NOT NULL PRIMARY KEY,
        fplName TEXT,
        name TEXT,
        playerGitId INTEGER
    )""")

    conn.commit()

def createPlayersTweetsTable(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE playersTweets (
        tweetId INTEGER NOT NULL PRIMARY KEY,
        playerId INTEGER,
        gameweek INTEGER,
        tweet TEXT
    )""")

def createPlayersTweetsInitialTeam(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE playersTweetsInitialTeam (
        tweetId INTEGER NOT NULL PRIMARY KEY,
        pID INTEGER,
        tweet TEXT,
        FOREIGN KEY (pID) REFERENCES players(playerId)
    )""")

    conn.commit()

def populatePlayersTweetsInitialTeam(conn):
    c = conn.cursor()
    with conn:
        c.execute("SELECT playerId, name FROM players")
    result = c.fetchall()
    
    fromDate = "2021-05-22"
    toDate = "2021-08-12"

    for tuple in result:
        worked = False
        id = tuple[0]
        name = tuple[1]
        c.execute("SELECT ictIndexSum FROM playersSumInitialTeam WHERE pID=?", (id,))
        try:
           ictIndexSum = c.fetchone()[0] 
           worked = True
        except:
            worked = False

        if (worked == True):
            limit = math.ceil(ictIndexSum)
            query = name + " lang:en until:" + toDate + " since:" + fromDate + " -filter:replies"
            tweets = retrieveTweets(query, limit)
            for tweet in tweets:
                with conn:
                    c.execute("INSERT INTO playersTweetsInitialTeam VALUES (NULL, ?, ?)", (id, tweet))

def populatePlayersTable(conn):
    c = conn.cursor()
    dataframe = pd.read_csv(PATH + '/2021-22/id_dict.csv')
    dataframe = dataframe.rename(columns={' FPL_ID': 'FPL_ID', ' FPL_Name': 'FPL_Name', ' Understat_Name': 'Understat_Name'})
    for i in dataframe.index:
        temp_id = int(dataframe.FPL_ID[i])
        fpl_name_without_accents = unidecode(dataframe.FPL_Name[i], errors='strict')
        name_without_accents = unidecode(dataframe.Understat_Name[i], errors='strict')
        with conn:
            c.execute("INSERT INTO players VALUES (NULL, ?, ?, ?)", (fpl_name_without_accents, name_without_accents, temp_id))

def populatePlayersTweetsTable(conn):
    c = conn.cursor()
    for i in range(37,38):
        dataframe1 = pd.read_csv(PATH + '/2021-22/gws/gw' + str(i) + '.csv')
        dataframe2 = pd.read_csv(PATH + '/2021-22/gws/gw' + str(i+1) + '.csv')
        for j in dataframe1.index:
            if dataframe1.minutes[j] > 0:
                limit = math.floor(dataframe1.influence[j]) * 10
                fromDate = dataframe1.kickoff_time[j][0:10]
                team = dataframe1.team[j]
                for k in dataframe2.index:
                    if dataframe2.team[k] == team:
                        toDate = dataframe2.kickoff_time[k][0:10]
                        break
                FPLName = unidecode(dataframe1.name[j], errors='strict')
                with conn:
                    c.execute("SELECT name FROM players WHERE FPLName=?", (FPLName,))
                    name = c.fetchone()[0]
                query = name + " lang:en until:" + toDate + " since:" + fromDate + " -filter:replies"
                tweets = retrieveTweets(query, limit)
                with conn:
                    c.execute("SELECT playerId FROM players WHERE FPLName=?", (FPLName,))
                    tempPlayerId = c.fetchone()[0]
                for tweet in tweets:
                    with conn:
                        c.execute("INSERT INTO playersTweets VALUES (NULL, ?, ?, ?)", (tempPlayerId, i, tweet))

def populatePlayersSumInitialTeamTable(conn):
    c = conn.cursor()
    dataframe1 = pd.read_csv(PATH + '/2021-22/id_dict.csv')
    dataframe1 = dataframe1.rename(columns={' FPL_ID': 'FPL_ID', ' FPL_Name': 'FPL_Name', ' Understat_Name': 'Understat_Name'})
    names = [dataframe1.FPL_Name[i] for i in dataframe1.index]

    for dirname in os.listdir('Fantasy-Premier-League/data/2020-21/players'):
        f = os.path.join('Fantasy-Premier-League/data/2020-21/players', dirname)

        i = len(f) - 1
        revid = ""

        while (f[i] != '_'):
            revid = revid + f[i]
            i-=1
        
        id = revid[::-1]

        dataframe2 = pd.read_csv(PATH + '/2020-21/player_idlist.csv')
        name = ""
        for i in dataframe2.index:
            if (str(dataframe2.id[i]) == str(id)):
                name = dataframe2.first_name[i] + " " + dataframe2.second_name[i]
                break

        if (name in names):
           dataframe3 = pd.read_csv(f+'/history.csv') 
           for i in dataframe3.index:
               if dataframe3.season_name[i] == "2020/21":
                   ict_index_value = dataframe3.ict_index[i]
                   break
        
        if (name in names):
            temp_name = unidecode(name, errors='strict')  
            with conn:
                c.execute("SELECT playerId FROM players WHERE FPLName=?", (temp_name,))
                tempPlayerId = c.fetchone()[0] 
                c.execute("INSERT INTO playersSumInitialTeam VALUES (?, ?)", (tempPlayerId, ict_index_value))

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    
    return conn

def retrieveTweets(q, l):
    tweetsCount = 0
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(q).get_items():
        if (tweetsCount == l):
            break
        else: 
            tweets.append(cleanTweet(tweet.content))
            tweetsCount+=1

    return tweets

#Expand Contractions
def expandContractions(text):
    return contractions.fix(text)

#Convert to Lower Case
def toLowerCase(text):
    return text.lower()

#Tokenization
def tokenization(text):
    return word_tokenize(text)

#Remove punctuation, symbols, numbers, dates, URLs, emojis and whitespace
def alphabetOnly(text):
    return [word for word in text if word.isalpha()]

#Remove stopwords
def removeStopwords(text):
    stopwords_nltk = set(stopwords.words('english'))
    return [word for word in text if word not in stopwords_nltk]

#Stemming
def stemming(text):
    porter = PorterStemmer()
    return [word for word in text if porter.stem(word)]



def cleanTweet(tweet):

    #new_text = expandContractions(tweet)
    lower_text = toLowerCase(tweet)
    tokenized_text = tokenization(lower_text)
    alphabetic_only = alphabetOnly(tokenized_text)
    without_stopwords = removeStopwords(alphabetic_only)
    stemmed_text = stemming(without_stopwords)

    return ' '.join(stemmed_text)

#NLTK Sentiment Analysis
def nltkSA(tweets):
    score = 0
    for tweet in tweets:
        polarity = SentimentIntensityAnalyzer().polarity_scores(tweet)
        score += polarity['compound']
    
    return score/len(tweets)

#Pattern Sentiment Analysis
def patternSA(tweets):
    score = 0
    for tweet in tweets:
        polarity = sentiment(tweet)
        score += polarity[0]
    
    return score/len(tweets)

#Pattern Sentiment Analysis
def textblobSA(tweets):
    score = 0
    for tweet in tweets:
        polarity = TextBlob(tweet).sentiment.polarity
        score += polarity
    
    return score/len(tweets)

#CoreNLP Sentiment Analysis
def coreNLPSA(tweets):
    score = 0
    for tweet in tweets:
        result = nlp.annotate(tweet,
                   properties={
                       'annotators': 'sentiment, ner, pos',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })
        
        for s in result["sentences"]:
            ogpolarity = s["sentimentValue"]
        
        polarity = (int(ogpolarity)/2)-1
        score += polarity
    
    return score/len(tweets)


def main():
    conn = create_connection('fpl.db')
    tweets = ["I'm not happy about this", "This is terrible", "What a bad day"]
    print("")
    ans = nltkSA(tweets)
    print("NLTK :" + str(ans))
    ans2 = patternSA(tweets)
    print("Pattern :" + str(ans2))
    ans3 = textblobSA(tweets)
    print("Textblob :" + str(ans3))
    ans4 = coreNLPSA(tweets)
    print("CoreNLP :" + str(ans4))
    print("Avg: " + str((ans+ans2+ans3+ans4)/4))

main()