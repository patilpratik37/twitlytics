import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv('tweets.csv')

# Instantiate sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define a function to assign sentiment scores to text
def get_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

# Apply the function to each row in the 'text' column and create a new column for sentiment scores
df['sentiment_score'] = df['text'].apply(get_sentiment)
df.head()

# Define a function to map continuous sentiment scores to binary classes
def get_sentiment_class(score, threshold=0.0):
    if score >= threshold:
        return 1  # Positive class
    else:
        return 0  # Negative class
    
# Apply the function to each row in the 'sentiment_score' column and create a new column for sentiment classes
df['sentiment_class'] = df['sentiment_score'].apply(get_sentiment_class)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment_class'], test_size=0.2, random_state=42)

# Train a Naive Bayes model on the training set
nb = MultinomialNB()
nb.fit(X_train_bow, y_train)

# Make predictions on the test set
y_pred = nb.predict(X_test_bow)

# Compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")