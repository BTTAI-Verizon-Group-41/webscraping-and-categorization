# ============================================
# 1. BERT as a Feature Extractor + XGBoost
# ============================================

# ----------------------------
# Import Necessary Libraries
# ----------------------------
import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from nltk.corpus import stopwords
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ----------------------------
# Download NLTK Data
# ----------------------------
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# ----------------------------
# Initialize NLP Tools
# ----------------------------
tokenizer_nltk = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
stop_words = set(stopwords.words('english'))

# ----------------------------
# Initialize BERT Tokenizer and Model
# ----------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.to(device)
bert_model.eval()  # Set model to evaluation mode

# ----------------------------
# Define Preprocessing Functions
# ----------------------------
def remove_placeholders(text):
    if pd.isna(text):
        return text
    # Remove emails
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '', text)
    # Remove credit card numbers or similar patterns
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
    # Remove redundant consecutive words
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
    # Remove HTML tags if present
    if '<' in text and '>' in text:
        text = BeautifulSoup(text, "html.parser").get_text()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def remove_stopwords_fn(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def calc_sentiment(text):
    return sia.polarity_scores(text)['compound'] if isinstance(text, str) else 0

def lexical_diversity(text):
    tokens = tokenize_text(text)
    return len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0

def tokenize_text(text):
    return tokenizer_nltk.tokenize(text) if isinstance(text, str) else []

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Apply all preprocessing steps
    text = remove_placeholders(text)
    text = remove_special_characters(text)
    text = remove_redundant(text)
    text = remove_curly_braces(text)
    text = remove_redundant_ngrams(text, n=2)
    text = remove_redundant_ngrams(text, n=3)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text).strip().lower()
    words = [word for word in text.split() if len(word) <= 25]
    tokens = tokenize_text(' '.join(words))
    return ' '.join(lemmatize_tokens(tokens))

# ----------------------------
# Function to Generate BERT Embeddings
# ----------------------------
def generate_bert_embeddings(texts, tokenizer, model, device, batch_size=32, max_length=128):
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating BERT embeddings"):
            batch_texts = texts[i:i+batch_size]
            encoded_input = tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use the [CLS] token representation
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
    return np.vstack(all_embeddings)

# ----------------------------
# Feature Engineering Function
# ----------------------------
def add_features_to(df, bert_features):
    # Preprocess text
    df['text_cleaned'] = df['content'].apply(preprocess_text)
    
    # Calculate sentiment and lexical diversity
    df['Sentiment'] = df['text_cleaned'].apply(calc_sentiment)
    df['Lexical_Diversity'] = df['text_cleaned'].apply(lexical_diversity)
    
    # TF-IDF Features
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text_cleaned'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
    tfidf_df = tfidf_df.add_prefix('TFIDF_')
    
    # BERT Features (Assuming PCA has been applied)
    bert_df = pd.DataFrame(bert_features, columns=[f'BERT_PCA_{i}' for i in range(bert_features.shape[1])])
    
    # Concatenate all features
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)
    return df

# ----------------------------
# Load and Preprocess Data
# ----------------------------
# Replace 'full.csv' with your actual data file path
df = pd.read_csv("full.csv")

# Apply feature engineering (excluding BERT for now)
df = add_features_to(df, bert_features=np.zeros((df.shape[0], 50)))  # Temporary placeholder

# ----------------------------
# Generate BERT Embeddings
# ----------------------------
texts = df['text_cleaned'].tolist()
bert_embeddings = generate_bert_embeddings(texts, bert_tokenizer, bert_model, device)

# ----------------------------
# Dimensionality Reduction with PCA
# ----------------------------
pca = PCA(n_components=50, random_state=42)  # Reduce to 50 dimensions
bert_reduced = pca.fit_transform(bert_embeddings)
print(f'Original BERT embeddings shape: {bert_embeddings.shape}')
print(f'Reduced BERT embeddings shape: {bert_reduced.shape}')

# ----------------------------
# Integrate BERT Features into DataFrame
# ----------------------------
df = add_features_to(df, bert_reduced)

# ----------------------------
# Prepare Labels
# ----------------------------
df['category'] = df['category'].astype(str)
df = df.dropna(subset=['category'])  # Ensure no missing labels

# Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['category'])

# ----------------------------
# Define Features and Labels
# ----------------------------
feature_columns = [col for col in df.columns if col.startswith('Sentiment') or
                   col.startswith('Lexical_Diversity') or
                   col.startswith('TFIDF_') or
                   col.startswith('BERT_PCA_')]
X = df[feature_columns]
y = y_encoded

# ----------------------------
# Split Data into Training and Testing Sets
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------
# Handle Class Imbalance
# ----------------------------
# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
sample_weights = class_weights[y_train]

# ----------------------------
# Initialize XGBClassifier
# ----------------------------
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),  # Ensure this matches your number of categories
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# ----------------------------
# Train the XGBClassifier with Sample Weights
# ----------------------------
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

# ----------------------------
# Evaluate on Test Set
# ----------------------------
y_pred = xgb_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
