# %% [markdown]
# # Week 1: Sales Forecasting
# ## Day 2: Data Engineering
# We will clean the dataset, resample it to monthly frequency, and create lag features.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv('../data/superstore_sales.csv', encoding='windows-1252')
print(df.head())

# %% [markdown]
# ### Clean and Preprocess Data
# The Superstore dataset has 'Order Date' and 'Sales'. We need to predict future sales.

# %%
# Convert 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# We'll forecast total sales, so let's group by Order Date and sum the Sales
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

# Set index to Order Date for resampling
daily_sales.set_index('Order Date', inplace=True)

# Resample to monthly sales (Start of Month)
monthly_sales = daily_sales['Sales'].resample('MS').sum()

# Plot the monthly sales
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o')
plt.title('Monthly Sales - Superstore')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# %% [markdown]
# ### Create Lag Features for XGBoost

# %%
# Create a dataframe for modeling
df_model = pd.DataFrame({'Sales': monthly_sales})

# Create Lag 1 and Lag 2 features (previous month's sales, and 2 months ago)
df_model['Lag_1'] = df_model['Sales'].shift(1)
df_model['Lag_2'] = df_model['Sales'].shift(2)

# Drop missing values caused by lagging
df_model.dropna(inplace=True)

print(df_model.head())

# %% [markdown]
# ## Day 3: Modeling
# We will train ARIMA and XGBoost models, and evaluate them.

# %%
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import pickle

# Train-Test Split (last 12 months as test set)
train_size = len(df_model) - 12
train, test = df_model.iloc[:train_size], df_model.iloc[train_size:]

print(f"Train size: {len(train)}, Test size: {len(test)}")

# %% [markdown]
# ### 1. ARIMA Model

# %%
# Train ARIMA on the training data
# Using a simple order (1, 1, 1) for demonstration
arima_model = ARIMA(train['Sales'], order=(1, 1, 1))
arima_result = arima_model.fit()

# Predict on test set
# Using steps = 12 (since our test set is 12 months)
arima_forecast = arima_result.forecast(steps=len(test))

# Evaluate
arima_rmse = root_mean_squared_error(test['Sales'], arima_forecast)
arima_mape = mean_absolute_percentage_error(test['Sales'], arima_forecast)

print(f"ARIMA RMSE: {arima_rmse:.2f}")
print(f"ARIMA MAPE: {arima_mape:.2f}")

# %% [markdown]
# ### 2. XGBoost Model

# %%
# Prepare features (X) and target (y)
X_train = train[['Lag_1', 'Lag_2']]
y_train = train['Sales']

X_test = test[['Lag_1', 'Lag_2']]
y_test = test['Sales']

# Train XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict
xgb_forecast = xgb_model.predict(X_test)

# Evaluate
xgb_rmse = root_mean_squared_error(y_test, xgb_forecast)
xgb_mape = mean_absolute_percentage_error(y_test, xgb_forecast)

print(f"XGBoost RMSE: {xgb_rmse:.2f}")
print(f"XGBoost MAPE: {xgb_mape:.2f}")

# %% [markdown]
# ### Compare and Save the Best Model
# Based on metrics, we will save the XGBoost model as it handles non-linear patterns well and we have lag features.

# %%
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Sales'], label='Train')
plt.plot(test.index, test['Sales'], label='Test (Actual)')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast')
plt.plot(test.index, xgb_forecast, label='XGBoost Forecast')
plt.legend()
plt.title('Model Forecast Comparison')
plt.show()

# %% [markdown]
# ## Day 4: Export Model

# %%
# Save XGBoost model to a pickle file
model_path = '../app/model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(xgb_model, f)
    
print(f"Model saved to {model_path}")
