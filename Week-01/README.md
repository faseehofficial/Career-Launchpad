# Week 1 - Sales Forecasting Project

This project predicts next month's product sales using a historical retail dataset. It covers the end-to-end process: Data Engineering, Modeling, and Deployment.

## Project Structure

- `data/`: Contains the Superstore Sales dataset.
- `notebooks/`: Contains the data engineering and modeling code (`01_EDA_and_Modeling.py`).
- `app/`: Contains the Flask API (`app.py`) and the trained XGBoost model (`model.pkl`).

## Setup

1. Create and activate the virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn xgboost flask plotly jupyter statsmodels
   ```

## Workflow

### 1. Data Collection
Run `download_data.py` to download the Superstore dataset from GitHub.

### 2. EDA & Modeling
Open `notebooks/01_EDA_and_Modeling.py`. You can run this file as an Interactive Notebook in VS Code. It will:
- Clean the data and resample it to monthly sales.
- Create lag features (`Lag_1`, `Lag_2`).
- Train ARIMA and XGBoost models.
- Export the best model (`model.pkl`) to the `app/` folder.

### 3. Deployment Dashboard
Run the Flask application:
```bash
cd app
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to interact with the Sales Forecasting Dashboard. It uses the exported model to predict future sales based on inputs.

## KPIs Used
- **RMSE (Root Mean Squared Error)**: Measures the average magnitude of the errors.
- **MAPE (Mean Absolute Percentage Error)**: Measures the error as a percentage of the actual values.

## Models Compared
- **ARIMA**: Statistical model capturing linear time-series trends.
- **XGBoost**: Tree-based machine learning model capturing complex non-linear patterns using lag features.
