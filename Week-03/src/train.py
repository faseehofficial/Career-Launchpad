import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import joblib
from preprocessing import prepare_data

def train_models():
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data("data/raw_german_credit.csv")
    
    print("\n--- Training XGBoost ---")
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(f"XGBoost ROC-AUC: {xgb_auc:.4f}")
    print(classification_report(y_test, xgb_preds))
    
    print("\n--- Training CatBoost ---")
    cat = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_state=42)
    cat.fit(X_train, y_train)
    cat_preds = cat.predict(X_test)
    cat_auc = roc_auc_score(y_test, cat.predict_proba(X_test)[:, 1])
    print(f"CatBoost ROC-AUC: {cat_auc:.4f}")
    print(classification_report(y_test, cat_preds))
    
    # Save the best model
    if cat_auc > xgb_auc:
        print("\nSaving CatBoost as the best model...")
        joblib.dump(cat, 'models/best_model.joblib')
        model_type = "CatBoost"
    else:
        print("\nSaving XGBoost as the best model...")
        joblib.dump(xgb, 'models/best_model.joblib')
        model_type = "XGBoost"
        
    return model_type

if __name__ == "__main__":
    train_models()
