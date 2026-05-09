from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Warning: model.pkl not found. Make sure to run the modeling script first.")

# Simple HTML template with Plotly.js for the dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sales Forecasting Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; }
        .container { max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .input-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold;}
        input { padding: 8px; width: 100%; box-sizing: border-box; }
        button { background-color: #007bff; color: white; border: none; padding: 10px 20px; cursor: pointer; border-radius: 4px; font-size: 16px; width: 100%;}
        button:hover { background-color: #0056b3; }
        #result { margin-top: 20px; font-size: 1.2em; font-weight: bold; text-align: center; color: #28a745; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Next Month Sales Forecast</h1>
        <div class="input-group">
            <label for="lag1">Sales 1 Month Ago ($):</label>
            <input type="number" id="lag1" value="50000" step="100">
        </div>
        <div class="input-group">
            <label for="lag2">Sales 2 Months Ago ($):</label>
            <input type="number" id="lag2" value="48000" step="100">
        </div>
        <button onclick="predict()">Get Forecast</button>
        
        <div id="result"></div>
        <div id="plot" style="margin-top: 30px;"></div>
    </div>

    <script>
        function predict() {
            const lag1 = document.getElementById('lag1').value;
            const lag2 = document.getElementById('lag2').value;
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ lag_1: parseFloat(lag1), lag_2: parseFloat(lag2) }),
            })
            .then(response => response.json())
            .then(data => {
                if(data.error) {
                    document.getElementById('result').innerText = "Error: " + data.error;
                    return;
                }
                const forecast = data.forecast;
                document.getElementById('result').innerText = `Forecasted Sales: $${forecast.toFixed(2)}`;
                
                // Plot
                const trace1 = {
                    x: ['2 Months Ago', '1 Month Ago', 'Next Month (Forecast)'],
                    y: [lag2, lag1, forecast],
                    type: 'scatter',
                    mode: 'lines+markers',
                    marker: {size: 12},
                    line: {color: '#007bff', width: 3}
                };
                const layout = { title: 'Sales Trend & Forecast' };
                Plotly.newPlot('plot', [trace1], layout);
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    data = request.json
    lag_1 = data.get('lag_1')
    lag_2 = data.get('lag_2')
    
    if lag_1 is None or lag_2 is None:
        return jsonify({'error': 'Please provide lag_1 and lag_2'}), 400
        
    # Prepare features for XGBoost
    features = pd.DataFrame({'Lag_1': [lag_1], 'Lag_2': [lag_2]})
    
    # Predict
    prediction = model.predict(features)[0]
    
    return jsonify({'forecast': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
