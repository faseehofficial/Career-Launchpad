import pandas as pd
import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def create_synthetic_data():
    """
    Creates a synthetic dataset representing social media posts for sentiment analysis.
    In a real scenario, this would involve Twitter API (Tweepy) or scraping.
    """
    data = {
        'text': [
            "I love the new features of this product! So amazing! #happy",
            "This service is absolutely terrible. I want a refund.",
            "Great experience today at the store. Staff was very helpful.",
            "I'm so frustrated with the constant crashes. Worst app ever.",
            "Neutral feeling about the update. Some good, some bad.",
            "Just had the best coffee of my life! @CoffeeShop #blessed",
            "The delivery was late and the box was damaged. Not happy.",
            "Really impressed with the customer support team. Fixed my issue in minutes!",
            "I don't understand the hype. It's just okay.",
            "Absolutely stunning visuals in the new movie. Highly recommend!"
        ],
        'sentiment': [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  # 1 for Positive, 0 for Negative
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, 'social_media_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Day 1 Task: Synthetic social media dataset created at {csv_path}")
    print(df.head())

if __name__ == "__main__":
    create_synthetic_data()
