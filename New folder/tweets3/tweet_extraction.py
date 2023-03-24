import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "Layoff OR Resignation OR Retrenchment OR Termination OR Turnover OR Workload OR Attrition OR Burnout OR Retention OR Fired OR Moonlighting #Infosys OR #TCS OR #Wipro lang:en until:2023-02-01 since:2020-02-01"
tweets = []
limit = 5000


for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.username, tweet.content])
        
df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
print(df)

# to save to csv
#df.to_csv('tweets.csv')
