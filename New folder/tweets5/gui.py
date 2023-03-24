import numpy as np
import pandas as pd
import gradio as gr
import warnings
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words ('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction. text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
warnings.filterwarnings('ignore')

df = pd.read_csv('tweets.csv')
df.head()
print (df.columns)

x = df.drop(['Unnamed: 0', 'Date', 'User'], axis=1)
x.head()
y = df['text']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y)

from sklearn.preprocessing import StandardScaler

#instantiate StandardScaler object
scaler = StandardScaler()

def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','',text)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

#scale data
x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.fit_transform(x_test)

#import model object
from sklearn.neural_network import MLPClassifier
model =  MLPClassifier(max_iter=1000,  alpha=1)

#train model on training data
model.fit(x_train_scaled, y_train)

#getting model performance on test data
print("accuracy:", model.score(x_test_scaled, y_test))

#geting our columns

print(df.columns)

def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin, BMI, Diabetes_Pedigree, Age):
#turning the arguments into a numpy array  

 x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])

  prediction = model.predict(x.reshape(1, -1))

  return prediction

outputs = gr.outputs.Textbox()

app = gr.Interface(fn=diabetes, inputs=['number','number','number','number','number','number','number','number'], outputs=outputs,description="This is a diabetes model")

app.launch()

#To provide a shareable link
app.launch(share=True)