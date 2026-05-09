import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

def get_preprocessing_pipeline(df):
    # Separate numeric and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features.remove('credit_risk') # Target
    
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    X = df.drop('credit_risk', axis=1)
    y = df['credit_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Get pipeline
    preprocessor = get_preprocessing_pipeline(df)
    
    # Fit and transform X_train
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)
    
    # Save the preprocessor for later use in the API
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    
    return X_train_res, X_test_processed, y_train_res, y_test

if __name__ == "__main__":
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    
    X_train, X_test, y_train, y_test = prepare_data("data/raw_german_credit.csv")
    print(f"Original training shape: {X_train.shape[0]} rows")
    print(f"Resampled training shape: {X_train.shape[0]} rows")
    print(f"Test shape: {X_test.shape[0]} rows")
    print("\nData preparation complete. Preprocessor saved to models/preprocessor.joblib")
