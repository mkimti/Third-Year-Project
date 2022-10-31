import snscrape.modules.twitter as sntwitter
import pandas as pd
import sqlite3
from sqlite3 import Error
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import contractions
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from nltk.corpus import words

PATH = 'Fantasy-Premier-League/data'

def createPlayersTable(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE players (
        playerId INTEGER NOT NULL PRIMARY KEY,
        firstname TEXT,
        lastname TEXT,
        playerGitId INTEGER
    )""")

    conn.commit()

def createPlayersTweetsTable(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE playersTweets (
        tweetId INTEGER NOT NULL PRIMARY KEY,
        FOREIGN KEY (playerId) REFERENCES players (playerId)
        gameweek INTEGER,
        tweet TEXT
    )""")


def populatePlayersTable(conn):
    c = conn.cursor()
    current_df = pd.read_csv(PATH + '/2021-22/player_idlist.csv')
    for i in current_df.index:
        temp_id = int(current_df.id[i])
        with conn:
            c.execute("INSERT INTO players VALUES (NULL, ?, ?, ?)", (current_df.first_name[i], current_df.second_name[i], temp_id))

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    
    return conn

def retrieveTweets(q, l):
    tweetsCount = 0
    for tweet in sntwitter.TwitterSearchScraper(q).get_items():
        if (tweetsCount == l):
            break
        else: 
            cleanTweet(tweet.content)
            tweetsCount+=1

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
    print(tweet) 

    new_text = expandContractions(tweet)
    lower_text = toLowerCase(new_text)
    tokenized_text = tokenization(lower_text)
    alphabetic_only = alphabetOnly(tokenized_text)
    without_stopwords = removeStopwords(alphabetic_only)
    stemmed_text = stemming(without_stopwords)

    print(stemmed_text)
    print(' '.join(stemmed_text))


def main():
    #query = "Ronaldo min_faves:500 lang:en"
    #limit = 1
    conn = create_connection('fpl.db')

    #retrieveTweets(query, limit)
    createPlayersTable(conn)
    populatePlayersTable(conn)

    #createPlayersTweetsTable(conn)



main()
