import pickle 
import pandas as pd
import seaborn as sns
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
import nltk.metrics
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from xgboost import XGBClassifier
import numpy as np
import re

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

sia = SentimentIntensityAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------------
# LOAD THE DATA
df = pd.read_csv('data.csv')

def calc_sentiment(text):
    if isinstance(text, str):
        return sia.polarity_scores(text)['compound']
    else:
        return 0

def tokenize_text(text):
    if isinstance(text, str):
        return tokenizer.tokenize(text)
    else:
        return []

def lemmatize_tokens(tokens):
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs and emails
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase
    text = text.lower()
    # Tokenize and lemmatize
    tokens = tokenize_text(text)
    lemmatized = lemmatize_tokens(tokens)
    # Rejoin tokens
    clean_text = ' '.join(lemmatized)
    return clean_text


def add_features_to(df):
    df['Text_Length'] = df['text_content'].str.len()
    df['Text_Length'] = df['Text_Length'].fillna(0)
    df['text_cleaned'] = df['text_content'].apply(preprocess_text)

    df['Sentiment'] = df['text_cleaned'].apply(calc_sentiment)
    return df

df_sentiment = add_features_to(df)
df_sentiment.to_csv('df_sentiment.csv', index = False)
df_text = df.dropna()
df_text.to_csv('df_text.csv', index = False)


non_string_count = df['text_content'].apply(lambda x: not isinstance(x, str)).sum()
print(f"Number of non-string entries in 'text_content': {non_string_count}")
print("Columns in the dataset:", df.columns.tolist())
print("Data types:\n", df.dtypes)
print(df_sentiment.head(10))
print((df['Text_Length'] == 0).sum())
print(df.shape)
print(df_text.head(20))
