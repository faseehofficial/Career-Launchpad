import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from collections import Counter
import re

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INSIGHTS_DIR = os.path.join(BASE_DIR, 'insights')

def generate_insights():
    input_path = os.path.join(DATA_DIR, 'cleaned_social_media_data.csv')
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run Day 2 script first.")
        return

    df = pd.read_csv(input_path)
    
    # Separate positive and negative texts
    positive_text = " ".join(df[df['sentiment'] == 1]['cleaned_text'])
    negative_text = " ".join(df[df['sentiment'] == 0]['cleaned_text'])
    
    os.makedirs(INSIGHTS_DIR, exist_ok=True)
    
    # Generate WordClouds
    def save_wordcloud(text, title, filename):
        if not text.strip():
            print(f"No text found for {title}")
            return
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        output_path = os.path.join(INSIGHTS_DIR, filename)
        plt.savefig(output_path)
        print(f"Insight saved: {output_path}")

    save_wordcloud(positive_text, "Positive Sentiment Keywords", "positive_wordcloud.png")
    save_wordcloud(negative_text, "Negative Sentiment Keywords", "negative_wordcloud.png")
    
    # Top Keywords Analysis
    def get_top_keywords(text, n=10):
        words = text.split()
        return Counter(words).most_common(n)

    print("\nTop 5 Positive Keywords:")
    print(get_top_keywords(positive_text, 5))
    
    print("\nTop 5 Negative Keywords:")
    print(get_top_keywords(negative_text, 5))
    
    print(f"\nDay 5 Task: Insights generated in {INSIGHTS_DIR} directory.")

if __name__ == "__main__":
    generate_insights()
