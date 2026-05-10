# Week-04: Sentiment Analysis (Social Media)

Goal: Detect positive/negative sentiment in social media posts using BERT.

## Project Structure
- `day1_data_collection.py`: Generates synthetic social media data (CSV).
- `day2_preprocessing.py`: Cleans text and demonstrates BERT tokenization.
- `day3_training.py`: Fine-tunes a DistilBERT model for sequence classification.
- `day4_api.py`: Serves the trained model via a FastAPI REST endpoint.
- `day5_insights.py`: Generates word clouds and extracts top business keywords.
- `requirements.txt`: Project dependencies.
- `data/`: Directory for raw and cleaned datasets.
- `models/`: Directory for saved model weights.
- `insights/`: Directory for visualization outputs.

## How to Run

### Step 0: Install Dependencies
Run this from the project root (`Career-Launchpad`):
```powershell
pip install -r Week-04/requirements.txt
```
Or navigate into the directory first:
```powershell
cd Week-04
pip install -r requirements.txt
```

### Day 1: Data Collection
```powershell
python Week-04/day1_data_collection.py
```

### Day 2: Text Preprocessing
```powershell
python Week-04/day2_preprocessing.py
```

### Day 3: Model Training
Fine-tune the BERT model.
```powershell
python Week-04/day3_training.py
```

### Day 4: Deployment (API)
Start the FastAPI server.
```powershell
python Week-04/day4_api.py
```
You can test the API using `curl` or by visiting `http://localhost:8000/docs` in your browser.
Example Request:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"I love this project!\"}"
```

### Day 5: Business Insights
Generate visualizations and keyword summaries.
```powershell
python Week-04/day5_insights.py
```

## Prerequisites
- Python 3.8+
- **Git LFS**: This project uses Git Large File Storage for the model weights (~255MB). 
  If you are cloning this repository, make sure you have it installed:
  ```bash
  git lfs install
  git lfs pull
  ```

## Insights Output
Check the `insights/` folder for `positive_wordcloud.png` and `negative_wordcloud.png`.
