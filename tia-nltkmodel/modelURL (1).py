import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder  # Added import for OneHotEncoder

# Initialize NLP tools
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
stop_words = set(stopwords.words('english'))
translator = GoogleTranslator(source='auto', target='en') 

def translate_text(text):
    if pd.isna(text):
        return text
    try:
        translated = translator.translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def extract_url_extension(url):
    """
    Extracts the file extension from a URL using regex.
    Returns 'no_extension' if no extension is found.
    """
    try:
        # Regex to find the extension after the last dot before end, query, or fragment
        match = re.search(r'\.([a-zA-Z0-9]+)(?:$|[?#])', url)
        return match.group(1).lower() if match else 'no_extension'
    except Exception as e:
        # In case of any parsing error, return 'invalid_url'
        return 'invalid_url'

# Preprocessing functions
def remove_placeholders(text):
    if pd.isna(text):
        return text
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '', text)
    text = re.sub(r'\b\d{4} \d{4} \d{4} \d{4}\b', '', text)
    return text

def remove_curly_braces(text):
    if pd.isna(text):
        return text
    return re.sub(r'\{.*?\}', '', text)

def remove_text_in_parentheses(text):
    if pd.isna(text):
        return text
    return re.sub(r'\(.*?\)', '', text)

def remove_redundant(text):
    if pd.isna(text):
        return text
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

def remove_redundant_ngrams(text, n=2):
    if pd.isna(text) or not isinstance(text, str):
        return text
    words = text.split()
    if len(words) < n:
        return text
    clean_words = []
    for i in range(len(words)):
        if i >= n and words[i-n:i] == words[i-n+1:i+1]:
            continue
        clean_words.append(words[i])
    return ' '.join(clean_words)

def remove_special_characters(text):
    if pd.isna(text):
        return text
    # Add a heuristic check or explicitly specify parser features to avoid warnings
    if '<' in text and '>' in text:  # Check for HTML-like content
        text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.strip()

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def calc_sentiment(text):
    return sia.polarity_scores(text)['compound'] if isinstance(text, str) else 0

def lexical_diversity(text):
    tokens = tokenize_text(text)
    return len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0

def tokenize_text(text):
    return tokenizer.tokenize(text) if isinstance(text, str) else []

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def pos_distribution(text):
    tokens = tokenize_text(text)
    tags = pos_tag(tokens)
    total = len(tags)
    counts = pd.Series([tag for _, tag in tags]).value_counts()
    return (counts / total).to_dict() if total > 0 else {}

def is_noisy_sentence(text, min_len=5, max_numeric_ratio=0.5):
    if pd.isna(text):
        return False
    words = text.split()
    num_ratio = sum(word.isdigit() for word in words) / len(words)
    return num_ratio > max_numeric_ratio or len(words) < min_len

# Updated preprocessing pipeline
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # text = translate_text(text) 
    text = remove_placeholders(text)
    text = remove_special_characters(text)
    text = remove_redundant(text)
    text = remove_text_in_parentheses(text)
    text = remove_curly_braces(text)
    text = remove_redundant_ngrams(text, n=2)
    text = remove_redundant_ngrams(text, n=3)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # text = translate_text(text) 
    text = re.sub(r'[^A-Za-z\s]', '', text).strip().lower()
    text = remove_stopwords(text) 
    words = [word for word in text.split() if len(word) <= 25]
    tokens = tokenize_text(' '.join(words))
    return ' '.join(lemmatize_tokens(tokens))

# Feature engineering function
def add_features_to(df):
    df['text_cleaned'] = df['content'].apply(preprocess_text)
    df['Sentiment'] = df['text_cleaned'].apply(calc_sentiment)
    df['Lexical_Diversity'] = df['content'].apply(lexical_diversity)
    df['url_extension'] = df['url'].apply(extract_url_extension)

    pos_features = df['text_cleaned'].apply(pos_distribution)
    pos_df = pd.DataFrame(pos_features.tolist()).fillna(0)
    pos_df = pos_df.add_prefix('POS_')
    df = pd.concat([df.reset_index(drop=True), pos_df.reset_index(drop=True)], axis=1)
    

    # One-Hot Encoding for 'url_extension'
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)  
    url_ext_encoded = encoder.fit_transform(df[['url_extension']])  
    url_ext_df = pd.DataFrame(url_ext_encoded, columns=encoder.get_feature_names(['url_extension']))
    df = pd.concat([df, url_ext_df], axis=1)
    df = df.drop(columns=['url_extension'])
    
    #TF-IDF Vectorization
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text_cleaned'])
    # Use get_feature_names for compatibility with older sklearn versions
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
    tfidf_df = tfidf_df.add_prefix('TFIDF_')
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    return df

# Load data
df = pd.read_csv("full.csv")
df = df.groupby('category').filter(lambda x: len(x) >= 2).reset_index(drop=True)
df = df[~df['content'].apply(is_noisy_sentence)].reset_index(drop=True)

# Identify categories with fewer than 2 occurrences

# Prepare data
df = add_features_to(df)
df['category'] = df['category'].astype(str)
df = df.dropna(subset=['category'])

# Define features and labels
X = df.drop(columns=['category', 'url', 'content', 'text_cleaned'], errors='ignore')
y = df['category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = XGBClassifier(n_estimators=75, learning_rate=0.12, max_depth=8, random_state=42, subsample=0.8, n_jobs=1)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save results
results_df = pd.DataFrame({
    "URL": df.loc[X_test.index, 'url'].values,
    "Actual": y_test.values,
    "Predicted": y_pred
})

# results_df.to_csv("xgbUrl.csv", index = False)

# Cross validation
cv_scores = cross_val_score(model, X, y, cv=2, scoring='accuracy', n_jobs=1)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
print("Standard Deviation of CV Accuracy:", np.std(cv_scores))


print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#RESULTS-----------------------

#model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
#Accuracy: 0.5140845070422535

#model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
#Accuracy: 0.5316901408450704

#model = XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=6, random_state=42)
#Accuracy: 0.5352112676056338

# Best Parameters: {'learning_rate': 0.12, 'max_depth': 4, 'n_estimators': 25}
# Test Accuracy: 0.5070422535211268

# Best Parameters: {'learning_rate': 0.12, 'n_estimators': 25, 'subsample': 0.8}
# Test Accuracy: 0.5316901408450704

# Best Parameters: {'learning_rate': 0.08, 'max_depth': 7, 'n_estimators': 25, 'subsample': 0.8}
# Test Accuracy: 0.5211267605633803

# Best Parameters: {'learning_rate': 0.12, 'max_depth': 8, 'n_estimators': 75, 'subsample': 0.8}
# Test Accuracy: 0.5563380281690141

# Cross-Validation Scores: [0.51391863 0.54389722]
# Mean CV Accuracy: 0.5289079229122056
# Standard Deviation of CV Accuracy: 0.01498929336188437
# Test Accuracy: 0.5373665480427047
