import snscrape.modules.twitter as sntwitter
import pandas as pd
import sqlite3
from sqlite3 import Error
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import contractions
import math
from unidecode import unidecode

PATH = 'Fantasy-Premier-League/data'

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
    for i in range(8,11):
        dataframe1 = pd.read_csv(PATH + '/2021-22/gws/gw' + str(i) + '.csv')
        dataframe2 = pd.read_csv(PATH + '/2021-22/gws/gw' + str(i+1) + '.csv')
        tempEarliestDay = int(dataframe2.kickoff_time[0][8:10])
        for k in dataframe2.index:
            if int(dataframe2.kickoff_time[k][8:10]) < tempEarliestDay:
                tempEarliestDay = int(dataframe2.kickoff_time[k][8:10])
        toDate = dataframe2.kickoff_time[0][0:8] + str(tempEarliestDay-1)
        for j in dataframe1.index:
            if dataframe1.minutes[j] > 0:
                limit = math.floor(dataframe1.influence[j]) * 10
                fromDate = dataframe1.kickoff_time[j][0:10]
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


def main():
    conn = create_connection('fpl.db')
    #createPlayersTable(conn)
    #populatePlayersTable(conn)

    #createPlayersTweetsTable(conn)
    populatePlayersTweetsTable(conn)



main()
