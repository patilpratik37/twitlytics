# Import necessary libraries
import pandas as pd
import numpy as np
import re
import string
import emoji
import num2words
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Load dataset
df = pd.read_csv('tweets.csv')
df.info()
df.tweet

# convert a column in lower case
df['tweet'] = df['tweet'].str.lower()
print (df)

# convert numerical value to words
def convert_to_words(num):
    return num2words(num)

# apply the function to the 'numbers' column
df['words'] = df['tweet'].apply(convert_to_words)

# print the resulting DataFrame
print(df)

# remove hyperlinks 
pattern = r'http\S+|www.\S+'
df['tweet'] = df['tweet'].apply(lambda x: re.sub(pattern, '', x))
print(df)

# remove hastags
pattern = r'#\w+'
df['tweet'] = df['tweet'].apply(lambda x: re.sub(pattern, '', x))
print(df)

# remove multilinguistic langugage and keep only english
pattern = r'[^\x00-\x7F]+'
df['tweet'] = df['tweet'].apply(lambda x: re.sub(pattern, '', x))
print(df)

# remove "@" mentions
pattern = r'@\w+'
df['tweet'] = df['tweet'].apply(lambda x: re.sub(pattern, '', x))
print(df)

# remove extra spaces
df['tweet'] = df['tweet'].apply(lambda x: ' '.join(x.split()))
print(df)

# remove operators
pattern = r'[^\w\s]'
df['tweet'] = df['tweet'].apply(lambda x: re.sub(pattern, '', x))
print(df)

# remove "_"
df['tweet'] = df['tweet'].apply(lambda x: x.replace('_', ''))
print(df)

# to save to csv
df.to_csv('tweets.csv')

# stemmer
stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

df['tweet'] = df['tweet'].apply(lambda x: stemming(x))

# polarity
def polarity(tweet):
    return TextBlob(tweet).sentiment.polarity

df['polarity'] = df['tweet'].apply(polarity)
     
     




















""""
# Remove all the rows which contain missing values
df_records_dropped = df.dropna(axis=0, how='any')
df_records_dropped.info()

# Drop only those records which contain has values missing from all the columns
df_records_dropped = df.dropna(axis=0, how='all')
df_records_dropped.info()
"""
"""
# Define function to preprocess tweets
def preprocess_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove URLs
    tweet = re.sub(r"http\S+", "", tweet)
    # Remove mentions
    tweet = re.sub(r"@\w+", "", tweet)
    # Remove hashtags
    tweet = re.sub(r"#\w+", "", tweet)
    # Remove punctuation
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace
    tweet = re.sub(r"\s+", " ", tweet).strip()
    # Replace emoticons with a special character
    tweet = re.sub(r"(:\)|:-\)|;\)|;-\)|:\(|:-\(|:'\(|:'-\()", "<emoticon>", tweet)

# Apply preprocessing function to 'text' column of dataframe
df['tweet'] = df['tweet'].apply(preprocess_tweet)

# Save preprocessed dataframe to CSV file
df.to_csv("preprocessed_tweets.csv", index=False)
"""

"""
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], test_size=0.3, random_state=40)

# Vectorize text data
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(max_depth=5, n_estimators=100, objective='multi:softmax', num_class=3)
xgb_model.fit(X_train_tfidf, y_train)

# Predict sentiment on testing set
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred = xgb_model.predict(X_test_tfidf)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
"""