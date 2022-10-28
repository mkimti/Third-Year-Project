import snscrape.modules.twitter as sntwitter
import pandas as pd

query = ""
tweets = []
limit = 100

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if (len(tweets) ==  limit):
        break
    else:
        tweets.append([tweet.date, tweet.content])

df = pd.DataFrame(tweets, columns=['Date', 'Tweet'])
print(df)