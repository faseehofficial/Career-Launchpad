from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os

app = FastAPI(title="Credit Scoring API", description="API to classify loan applicant risk level")

# Load model and preprocessor
MODEL_PATH = "models/best_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"

if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
else:
    model = None
    preprocessor = None

class ApplicantData(BaseModel):
    status: str
    duration: int
    credit_history: str
    purpose: str
    amount: int
    savings: str
    employment_duration: str
    installment_rate: int
    personal_status_sex: str
    other_debtors: str
    present_residence: int
    property: str
    age: int
    other_installment_plans: str
    housing: str
    number_credits: int
    job: str
    people_liable: int
    telephone: str
    foreign_worker: str

@app.get("/")
def read_root():
    return {"message": "Credit Scoring API is running. Use /predict to get risk classification."}

@app.post("/predict")
def predict_risk(data: ApplicantData):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not trained or not found.")
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Preprocess
    processed_data = preprocessor.transform(input_df)
    
    # Predict
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0].tolist()
    
    risk_level = "Good" if prediction == 1 else "Bad"
    
    return {
        "risk_classification": risk_level,
        "probability_good": probability[1],
        "probability_bad": probability[0]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
