import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


import warnings
warnings.filterwarnings("ignore")

nltk.download("stopwords")

df=pd.read_csv("email.csv")

df['Category']=df['Category'].map({'ham':0,'spam':1})
df=df.dropna(subset=['Category'])

ps=PorterStemmer()
stop_word=set(stopwords.words('english'))

def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z0-9]'," ",text)
    words=text.split()
    words=[ps.stem(word) for word in words if word not in stop_word ]
    return " ".join(words)

df['clean_meassage']=df['Message'].apply(clean_text)

x=df['clean_meassage']
y=df['Category']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

tf=TfidfVectorizer(max_features=5000)
x_train_tf=tf.fit_transform(x_train)
x_test_tf=tf.transform(x_test)

model=MultinomialNB()
model.fit(x_train_tf,y_train)

y_pred=model.predict(x_test_tf)

def predict_spam(message):
    cleaned=clean_text(message)
    vector=tf.transform([cleaned])
    r=model.predict(vector)[0]

    if r==1:
        return 'Spam'
    else:
        return 'Not Spam'


st.title("📩 Spam Message Detector")
st.write("Type a message below and check whether it is Spam or Not Spam.")

user_input=st.text_area("Enter Your Message Here")

if st.button('Predict'):
    if user_input.strip()=="":
         st.warning("Please enter a message first.")
    else:
        prediction = predict_spam(user_input)
        st.subheader("Result:")
        st.success(prediction)
        