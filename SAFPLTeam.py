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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from numpy import exp
import numpy as np
import pulp


PATH = 'Fantasy-Premier-League/data'

def createSquadsNormal(conn):
    c = conn.cursor()
    
    c.execute("""CREATE TABLE squadsNormal (
        playerId INTEGER,
        playerGitId INTEGER,
        name TEXT,
        gw1 INTEGER,
        gw2 INTEGER,
        gw3 INTEGER,
        gw4 INTEGER,
        gw5 INTEGER,
        gw6 INTEGER,
        gw7 INTEGER,
        gw8 INTEGER,
        gw9 INTEGER,
        gw10 INTEGER,
        gw11 INTEGER,
        gw12 INTEGER,
        gw13 INTEGER,
        gw14 INTEGER,
        gw15 INTEGER,
        gw16 INTEGER,
        gw17 INTEGER,
        gw18 INTEGER,
        gw19 INTEGER,
        gw20 INTEGER,
        gw21 INTEGER,
        gw22 INTEGER,
        gw23 INTEGER,
        gw24 INTEGER,
        gw25 INTEGER,
        gw26 INTEGER,
        gw27 INTEGER,
        gw28 INTEGER,
        gw29 INTEGER,
        gw30 INTEGER,
        gw31 INTEGER,
        gw32 INTEGER,
        gw33 INTEGER,
        gw34 INTEGER,
        gw35 INTEGER,
        gw36 INTEGER,
        gw37 INTEGER,
        gw38 INTEGER,
        FOREIGN KEY (playerId) REFERENCES players(playerId)
    )""")

    conn.commit()

def createPlayersSA(conn):
    c = conn.cursor()
    
    c.execute("""CREATE TABLE playersSANormal (
        pID INTEGER,
        name TEXT,
        nltk INTEGER,
        pattern INTEGER,
        textblob INTEGER,
        corenlp INTEGER,
        bert INTEGER,
        avg INTEGER,
        score INTEGER,
        gameweek INTEGER,
        FOREIGN KEY (pID) REFERENCES players(playerId)
    )""")

    conn.commit()

def createPlayersSumInitialTeam(conn):
    c = conn.cursor()
    
    c.execute("""CREATE TABLE playersSumInitialTeam (
        pID INTEGER,
        ictIndexSum INTEGER,
        FOREIGN KEY (pID) REFERENCES players(playerId)
    )""")

    conn.commit()

def createBoughtValueNormal(conn):
    c = conn.cursor()
    
    c.execute("""CREATE TABLE boughtValueNormal (
        playerId INTEGER,
        playerGitId INTEGER,
        name TEXT,
        boughtValue INTEGER,
        FOREIGN KEY (playerId) REFERENCES players(playerId)
    )""")

    conn.commit()

def createPlayersIctIndex(conn):
    c = conn.cursor()
    
    c.execute("""CREATE TABLE playersIctIndex (
        pID INTEGER,
        ictIndex INTEGER,
        gameweek INTEGER,
        FOREIGN KEY (pID) REFERENCES players(playerId)
    )""")

    conn.commit()

def createPerformanceNormal(conn):
    c = conn.cursor()
    
    c.execute("""CREATE TABLE performanceNormal (
        budget INTEGER,
        totalPoints INTEGER,
        gameweek INTEGER
    )""")

    conn.commit()

def createPlayersInitialSA(conn):
    c = conn.cursor()
    
    c.execute("""CREATE TABLE playersInitialSA (
        SAId INTEGER NOT NULL PRIMARY KEY,
        pID INTEGER,
        name TEXT,
        nltk INTEGER,
        pattern INTEGER,
        textblob INTEGER,
        corenlp INTEGER,
        bert INTEGER,
        avg INTEGER,
        FOREIGN KEY (pID) REFERENCES players(playerId)
    )""")

    conn.commit()

def createPlayersTable(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE players (
        playerId INTEGER NOT NULL PRIMARY KEY,
        fplName TEXT,
        name TEXT,
        playerGitId INTEGER,
        position INTEGER,
        team INTEGER,
        totalScore INTEGER
    )""")

    conn.commit()

def createPlayersTweetsNormal(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE playersTweetsNormal (
        tweetId INTEGER NOT NULL PRIMARY KEY,
        playerId INTEGER,
        gameweek INTEGER,
        tweet TEXT
    )""")

def createPlayersTweetsInitialTeamNormal(conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE playersTweetsInitialTeamNormal (
        tweetId INTEGER NOT NULL PRIMARY KEY,
        pID INTEGER,
        tweet TEXT,
        FOREIGN KEY (pID) REFERENCES players(playerId)
    )""")

    conn.commit()

def populatePlayersSA(conn):
    c = conn.cursor()
    with conn:
        c.execute("SELECT * FROM players")
    result = c.fetchall()

    for tuple in result:
        worked = False
        id = tuple[0]
        name = tuple[1]
        score = tuple[6]
        
        with conn:
            c.execute("SELECT * FROM playersInitialSA WHERE pID = ?", (id,))
        result = c.fetchall()

        if (len(result)) == 0:
            with conn:
                c.execute("INSERT INTO playersSA VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (id, name, 0, 0, 0, 0, 0, 0, score, 1, "normal"))
        else:
            nltk = result[0][3]
            pattern = result[0][4]
            textblob = result[0][5]
            corenlp = result[0][6]
            bert = result[0][7]
            avg = result[0][8] 

            with conn:
                c.execute("INSERT INTO playersSA VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (id, name, nltk, pattern, textblob, corenlp, bert, avg, score, 1, "normal"))

def findingPenalty():
    total = 0
    for i in range(1, 39):
        dataframe = pd.read_csv(PATH + '/2021-22/gws/gw' + str(i) + '.csv')
        points = [dataframe.total_points[k] for k in dataframe.index]
        count = 0
        for j in range(len(points)):
            if points[j] >= 4:
                count+=1
        
        fraction = count/len(points)
        print("GW: " + str(i))
        print("Count: " + str(count))
        print("Total: " + str(len(points)))
        print("Percentage: " + str(math.ceil(fraction*100)) + "%")
        total += fraction
    
    print(str(math.ceil((total/38)*100)) + "%")

def populateSquadsNormal(conn):
    c = conn.cursor()
    with conn:
        c.execute("SELECT playerId, name, playerGitId FROM players")
    result = c.fetchall()

    for tuple in result:
        playerId = tuple[0]
        name = tuple[1]
        playerGitId = tuple[2]

        with conn:
            c.execute("INSERT INTO squadsNormal VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)", (playerId, playerGitId, name))

def populateBoughtValueNormal(conn):
    c = conn.cursor()
    with conn:
        c.execute("SELECT playerId, name, playerGitId FROM players")
    result = c.fetchall()

    for tuple in result:
        playerId = tuple[0]
        name = tuple[1]
        playerGitId = tuple[2]

        with conn:
            c.execute("INSERT INTO boughtValueNormal VALUES (?, ?, ?, -1)", (playerId, playerGitId, name))

def populatePlayersInitialSA(conn):
    c = conn.cursor()
    with conn:
        c.execute("SELECT playerId, name FROM players")
    result = c.fetchall()

    for tuple in result:
        worked = False
        id = tuple[0]
        name = tuple[1]
        c.execute("SELECT tweet FROM playersTweetsInitialTeam WHERE pID=?", (id,))
        tweets = c.fetchall()

        if (len(tweets) != 0):
            finalTweets = []
            for tweet in tweets:
                finalTweets.append(tweet[0])
            nltkScore = nltkSA(finalTweets)
            patternScore = patternSA(finalTweets)
            textblobScore = textblobSA(finalTweets)
            corenlpScore = coreNLPSA(finalTweets)
            bertScore = bert(finalTweets)
            average = np.round((nltkScore+patternScore+textblobScore+corenlpScore+bertScore)/5,4)
            with conn:
                c.execute("INSERT INTO playersInitialSA VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)", (id, name, nltkScore, patternScore, textblobScore, corenlpScore, bertScore, average))

def populatePlayersIctIndex(conn):
    c = conn.cursor()
    with conn:
        c.execute("SELECT playerId, fplName FROM players")
    result = c.fetchall()

    for i in range(1, 39):
        dataframe = pd.read_csv(PATH + '/2021-22/gws/gw' + str(i) + '.csv')
        for tuple in result:
            playerId = tuple[0]
            name = tuple[1]
            worked = False
            for j in dataframe.index:
                if (unidecode(dataframe.name[j], errors='strict') == name):
                    worked = True
                    ict_index = dataframe.ict_index[j]
                    break
            
            if (worked == False):
                ict_index = 0
            
            with conn:
                c.execute("INSERT INTO playersIctIndex VALUES (?, ?, ?)", (playerId, ict_index, i))



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

def getTotalScore(conn):
    c = conn.cursor()
    with conn:
        c.execute("SELECT playerId, name FROM players")
    result = c.fetchall()

    c.execute("SELECT ictIndexSum FROM playersSumInitialTeam ORDER BY ictIndexSum DESC LIMIT 1")
    maxIctIndex = c.fetchone()[0]

    c.execute("SELECT ictIndexSum FROM playersSumInitialTeam ORDER BY ictIndexSum ASC LIMIT 1")
    minIctIndex = c.fetchone()[0]

    for tuple in result:
        workedAvg = False
        id = tuple[0]
        name = tuple[1]
        c.execute("SELECT avg FROM playersInitialSA WHERE pID=?", (id,))
        try:
           avg = c.fetchone()[0] 
           workedAvg = True
        except:
            workedAvg = False
        
        if (workedAvg == False):
            avg = 0
        
        workedICT = False
        c.execute("SELECT ictIndexSum FROM playersSumInitialTeam WHERE pID=?", (id,))
        try:
           ictIndex = c.fetchone()[0]
           workedICT = True
        except:
            workedICT = False
        
        if (workedICT == False):
            ictIndex = 0
        
        if (ictIndex < 10):
            totalScore = 0
        else:
            normIctIndex = (ictIndex - minIctIndex) / (maxIctIndex - minIctIndex)
            totalScore = math.ceil(((normIctIndex * 0.3) + (avg * 0.7))*1000)

        with conn:
            c.execute("UPDATE players SET totalScore = ? WHERE playerId = ?", (totalScore, id))

def buildInitialTeamNormal(conn):
    truePositions = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
    dataframe = pd.read_csv(PATH + '/2021-22/players_raw.csv')
    ids = [int(dataframe.id[i]) for i in dataframe.index]
    positions = []
    teams = []
    names = []
    scores = []
    newIds = []
    costs = []

    c = conn.cursor()

    for id in ids:
        with conn:
            c.execute("SELECT fplName, team, position, playerId FROM players WHERE playerGitId = ?" ,(id,))
            result = c.fetchall()
        
        if (len(result) != 0):
            name = result[0][0]
            team = result[0][1]
            position = result[0][2]
            playerId = result[0][3]
            names.append(name)
            teams.append(team)
            positions.append(position)
            newIds.append(id)

            with conn:
                c.execute("SELECT score FROM playersSANormal WHERE pID = ? AND gameweek = ?" ,(playerId, 1))
                result2 = c.fetchall()
            
            score = result2[0][0]
            scores.append(score)
    
            for i in dataframe.index:
                if int(dataframe.id[i]) == id:
                    costs.append(int(dataframe.now_cost[i]))
                    break
    
    #Define LP Model
    model = pulp.LpProblem("Building_Initial_FPL_Team", pulp.LpMaximize)

    #Define Variables
    in_squad_choice = []
    for i in range(len(names)):
        in_squad_choice.append(pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer'))
    
    #Define Objective
    model += sum(in_squad_choice[i] * scores[i] for i in range(len(names))), "Objective"

    #Monetary Constraints
    model += sum(in_squad_choice[i] * costs[i] for i in range(len(names))) <= 1000

    #Squad Constraints
    model += sum(in_squad_choice) == 15.0

    model += sum(in_squad_choice[i] for i in range(len(names)) if positions[i] == 1) == 2.0
    model += sum(in_squad_choice[i] for i in range(len(names)) if positions[i] == 2) == 5.0
    model += sum(in_squad_choice[i] for i in range(len(names)) if positions[i] == 3) == 5.0
    model += sum(in_squad_choice[i] for i in range(len(names)) if positions[i] == 4) == 3.0

    teamsUniqueSet = set(teams)
    teamsUnique = list(teamsUniqueSet)

    for teamId in teamsUnique:
        model += sum(in_squad_choice[i] for i in range(len(names)) if teams[i] == teamId) <= 3.0
   
    #Solve LP Problem
    model.solve()
    #total_cost = 0
    #for i in range(len(names)):
        #if (in_squad_choice[i].value() == 1.0):
            #print("Name: " + names[i])
            #print("Score: " + str(scores[i]))
            #print("Cost: " + str(costs[i]))
            #print("Position: " + truePositions[positions[i]-1])
            #print("-------------------------------")
            #total_cost += costs[i]
    
    #print("Total Spent: " + str(total_cost))
    #print("Budget Remaining for Next Week: " + str(1000-total_cost))
    total_cost = 0
    for i in range(len(names)):
        if (in_squad_choice[i].value() == 1.0):
            id = newIds[i]
            cost = costs[i]
            #with conn:
                #c.execute("UPDATE squadsNormal SET gw1 = 1 WHERE playerGitId = ?", (id,))
                #c.execute("UPDATE boughtValueNormal SET boughtValue = ? WHERE playerGitId = ?", (cost, id))
            total_cost += costs[i]
    
    budget = 1000-total_cost
    with conn:
        c.execute("INSERT INTO performanceNormal VALUES (?, 0, 1)", (budget,))
        

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
    
    return np.round(score/len(tweets),4)

#Pattern Sentiment Analysis
def textblobSA(tweets):
    score = 0
    for tweet in tweets:
        polarity = TextBlob(tweet).sentiment.polarity
        score += polarity
    
    return np.round(score/len(tweets),4)

#CoreNLP Sentiment Analysis
def coreNLPSA(tweets):
    score = 0
    for tweet in tweets:
        result = nlp.annotate(tweet,
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })
        
        for s in result["sentences"]:
            ogpolarity = s["sentimentValue"]
        
        polarity = (int(ogpolarity)/2)-1
        score += polarity
    
    return np.round(score/len(tweets),4)

#Softmax Activation Function used in BERT
def softmax(vector):
    e = exp(vector)
    return e/e.sum()

#BERT Sentiment Analysis
def bert(tweets):
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    score = 0

    for tweet in tweets:
        encoded_input = tokenizer(tweet, return_tensors='pt')
        output = model(**encoded_input)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        if (config.id2label[ranking[0]] == 'positive'):
            polarity = np.round(float(scores[ranking[0]]), 4)
        elif(config.id2label[ranking[0]] == 'negative'):
            polarity = np.round(-float(scores[ranking[0]]), 4)
        else:
            if (config.id2label[ranking[1]] == 'positive'):
                polarity = np.round(float(scores[ranking[1]]), 4)
            else:
                polarity = 0-np.round(float(scores[ranking[1]]), 4)
        
        score += polarity
    
    return np.round(score/len(tweets),4)

def updatePlayersTable(conn):
    dataframe = pd.read_csv(PATH + '/2021-22/players_raw.csv')
    c = conn.cursor()

    with conn:
        c.execute("SELECT playerId, playerGitId FROM players")
        result = c.fetchall()
    
    for tuple in result:
        playerId = tuple[0]
        playerGitId = tuple[1]

        worked = False
        
        for i in dataframe.index:
            if int(dataframe.id[i]) == playerGitId:
                worked = True
                position = int(dataframe.element_type[i])
                team = int(dataframe.team[i])
                break
        
        if (worked == True):
            with conn:
                c.execute("UPDATE players SET position = ?, team = ? WHERE playerId = ?", (position, team, playerId))
        else:
            print("issue with: " + playerId)

        



def main():
    conn = create_connection('fpl.db')
    buildInitialTeamNormal(conn)
    #populateBoughtValueNormal(conn)
    #createPerformanceNormal(conn)


main()