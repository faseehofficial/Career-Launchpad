import pandas as pd
import re
from transformers import AutoTokenizer
import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions (@user)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (#tag) - keeping the word
    text = re.sub(r'#', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower().strip()
    return text

def preprocess_data():
    input_path = os.path.join(DATA_DIR, 'social_media_data.csv')
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run Day 1 script first.")
        return

    df = pd.read_csv(input_path)
    print("Original Data Sample:")
    print(df['text'].head())

    # Apply cleaning
    df['cleaned_text'] = df['text'].apply(clean_text)
    print("\nCleaned Data Sample:")
    print(df['cleaned_text'].head())

    # Tokenization using BERT (DistilBERT)
    print("\nTokenizing using DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Example of tokenization for the first row
    tokens = tokenizer(df['cleaned_text'].tolist(), padding=True, truncation=True, return_tensors="pt")
    
    # Save cleaned data
    output_path = os.path.join(DATA_DIR, 'cleaned_social_media_data.csv')
    df.to_csv(output_path, index=False)
    print(f"\nDay 2 Task: Data cleaned and saved to {output_path}")
    print(f"Tokenized shape: {tokens['input_ids'].shape}")

if __name__ == "__main__":
    preprocess_data()
