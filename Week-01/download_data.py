import pandas as pd
import os

url = "https://raw.githubusercontent.com/datasets/sourcing-sample-superstore/master/Sample%20-%20Superstore.csv"

def download_data():
    print(f"Downloading dataset from {url}...")
    try:
        # The superstore dataset often uses windows-1252 encoding
        df = pd.read_csv(url, encoding='windows-1252')
        output_path = os.path.join('data', 'superstore_sales.csv')
        df.to_csv(output_path, index=False)
        print(f"Dataset downloaded successfully and saved to {output_path}")
        print(f"Shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Failed to download dataset. Error: {e}")

if __name__ == "__main__":
    download_data()
