import pandas as pd
import numpy as np

def run_eda(file_path):
    df = pd.read_csv(file_path)
    
    print("--- Dataset Overview ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    print("\n--- Class Distribution ---")
    print(df['credit_risk'].value_counts(normalize=True))
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Unique Values in Categorical Columns ---")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        print(f"{col}: {df[col].nunique()} unique values")

if __name__ == "__main__":
    run_eda("data/raw_german_credit.csv")
