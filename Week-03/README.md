# Week 03: Credit Scoring (Bank Loan Risk)

## Goal
Classify risk level for new loan applicants to assist in bank underwriting decisions.

## Day 1: Understanding Underwriting Rules

Underwriting is the process by which a lender evaluates the risk of lending money to a borrower. In credit scoring, the "5 C's of Credit" are traditionally used:

1.  **Character**: The borrower's reputation or track record for repaying debts. This is often represented by **Credit History**.
2.  **Capacity**: The borrower's ability to repay the loan. Key metrics include **Debt-to-Income (DTI) Ratio** and income stability.
3.  **Capital**: The amount of money the borrower has. This includes savings, investments, or other assets that could be used to repay the loan.
4.  **Collateral**: Assets that the borrower offers as security for the loan. If the borrower defaults, the lender can seize the collateral.
5.  **Conditions**: The terms of the loan (interest rate, amount) and external factors like the state of the economy.

### Key Features for Modeling
- **Age**: Younger or very old applicants might have different risk profiles.
- **Income**: Total annual income.
- **Home Ownership**: Rent, Own, Mortgage.
- **Loan Amount**: Total amount requested.
- **Loan Intent**: Purpose of the loan (e.g., Education, Medical, Venture, Personal).
- **Loan Grade**: Pre-assigned risk grade (if available).
- **Historical Default**: Whether the borrower has defaulted before.
- **Credit History Length**: Number of years since first credit line.

## Project Structure
- `data/`: Raw and processed datasets.
- `notebooks/`: Jupyter notebooks for EDA and modeling.
- `src/`: Source code for feature engineering, training, and reporting.
- `app/`: FastAPI application for deployment.
- `models/`: Saved model and preprocessor.
- `reports/`: Generated reports and visualizations.

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
To process data, train the model, and generate the report:
```bash
python src/preprocessing.py
python src/train.py
python src/executive_report.py
```

### 3. Start the API
To deploy the model via FastAPI:
```bash
uvicorn app.main:app --reload
```

### 4. Test the API
You can send a POST request to `http://localhost:8000/predict` with the following JSON format:
```json
{
  "status": "0 <= ... < 200 DM",
  "duration": 24,
  "credit_history": "existing credits paid back duly till now",
  "purpose": "radio/television",
  "amount": 3000,
  "savings": "... < 100 DM",
  "employment_duration": "1 <= ... < 4 years",
  "installment_rate": 3,
  "personal_status_sex": "male : single",
  "other_debtors": "none",
  "present_residence": 4,
  "property": "car or other",
  "age": 35,
  "other_installment_plans": "none",
  "housing": "own",
  "number_credits": 1,
  "job": "skilled employee/official",
  "people_liable": 1,
  "telephone": "yes",
  "foreign_worker": "yes"
}
```
