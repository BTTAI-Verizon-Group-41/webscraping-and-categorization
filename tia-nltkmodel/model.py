# Import necessary packages
import pickle 
import pandas as pd
# import seaborn as sns
from os.path import exists
# import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import nltk
import nltk.metrics
from nltk import ngrams
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
# from xgboost import XGBClassifier
import numpy as np
import re
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from sklearn.naive_bayes import MultinomialNB



# Set up NLP tools
# nltk.download('vader_lexicon') # Run once
# nltk.download('wordnet') # Run once
sia = SentimentIntensityAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Load the data
df1 = pd.read_csv('data.csv') # Tia
# with open('data.csv', 'r', encoding='utf-8', errors='ignore') as file: # Athena
#     df1 = pd.read_csv(file) # Athena
df2 = pd.read_csv('keywords_emptyText.csv') # Tia
# with open('keywords_emptyText.csv', 'r', encoding='utf-8', errors='ignore') as file: # Athena
#     df2 = pd.read_csv(file) # Athena
df = df1[~df1['url'].isin(df2['url'])]

# Define text processing functions
def calc_sentiment(text):
    return sia.polarity_scores(text)['compound'] if isinstance(text, str) else 0

def tokenize_text(text):
    return tokenizer.tokenize(text) if isinstance(text, str) else []

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def lexical_diversity(text):
    tokens = tokenize_text(text)
    return len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0

def preprocess_text(text):
    if not isinstance(text, str):  # Check if text is not a string
        return ""
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text).strip().lower()
    tokens = tokenize_text(text)
    return ' '.join(lemmatize_tokens(tokens))

# Define LDA function to extract topics as features
def apply_lda(df, num_topics=10):
    tokenized_text = df['text_cleaned'].apply(tokenize_text)
    dictionary = Dictionary(tokenized_text)
    corpus = [dictionary.doc2bow(text) for text in tokenized_text]
    
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    topic_features = []
    for doc in corpus:
        topic_distribution = lda_model.get_document_topics(doc, minimum_probability=0)
        topic_probs = [prob for _, prob in sorted(topic_distribution, key=lambda x: x[0])]
        topic_features.append(topic_probs)
    
    topic_df = pd.DataFrame(topic_features, columns=[f"Topic_{i}" for i in range(num_topics)])
    topic_df = topic_df.add_prefix('LDA_')  # Add prefix to avoid column name conflicts
    return topic_df


def add_features_to(df):
    df = df.copy()
    df['Text_Length'] = df['text_content'].str.len().fillna(0)
    df['text_cleaned'] = df['text_content'].apply(preprocess_text)
    df['Sentiment'] = df['text_cleaned'].apply(calc_sentiment)
    df['lexical_diversity'] = df['text_content'].apply(lexical_diversity)
    
    # Calculate TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text_cleaned'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()) # fixed: get_feature_names
    tfidf_df = tfidf_df.add_prefix('TFIDF_')
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    
    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]
    return df



# Prepare the data with new features

df_new = add_features_to(df)
print(df_new['category'].head(15))
df_new['category'] = df_new['category'].astype(str)

# Ensure there are no missing labels
df_new = df_new.dropna(subset=['category'])

# Define features (X) and labels (y)
X = df_new.drop(columns=['category', 'url', 'text_content', 'text_cleaned'], errors='ignore')  # Features
y = df_new['category']  # Labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

results_df = pd.DataFrame({
    "URL": df_new.loc[X_test.index, 'url'].values,
    "Actual": y_test.values,
    "Predicted": y_pred
})
# Show misclassified samples
misclassified = results_df[results_df["Actual"] != results_df["Predicted"]]
misclassified.to_csv('comparison_mismatch.csv', index=False)

# Save correctly classified samples
correctly_classified = results_df[results_df["Actual"] == results_df["Predicted"]]
correctly_classified.to_csv('comparison_match.csv', index=False)











#lda implentation  w accuracyc 11%
# def add_features_to(df):
#     df = df.copy()  # Avoid SettingWithCopyWarning
#     df['Text_Length'] = df['text_content'].str.len().fillna(0)
#     df['text_cleaned'] = df['text_content'].apply(preprocess_text)
#     df['Sentiment'] = df['text_cleaned'].apply(calc_sentiment)
#     df['lexical_diversity'] = df['text_content'].apply(lexical_diversity)
    
#     # Calculate TF-IDF
#     tfidf_matrix = tfidf_vectorizer.fit_transform(df['text_cleaned'])
#     tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
#     tfidf_df = tfidf_df.add_prefix('TFIDF_')  # Add prefix to avoid column name conflicts
#     df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    
#     # Apply LDA to extract topic features
#     topic_df = apply_lda(df, num_topics=10)
#     df = pd.concat([df.reset_index(drop=True), topic_df.reset_index(drop=True)], axis=1)
#     return df


#logistic regression
#30% ACCURACY