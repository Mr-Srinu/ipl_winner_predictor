# IPL Winner Predictor

This project trains a machine learning model to predict IPL match winners using the Kaggle dataset
**ipl-complete-dataset-2008-2024** via [kagglehub](https://github.com/Kaggle/kagglehub).

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model (downloads data automatically via KaggleHub):
```bash
python train.py
```

3. Run the Streamlit web app:
```bash
streamlit run app.py
```

## Dataset

We use the Kaggle dataset `patrickb1912/ipl-complete-dataset-20082020` which contains:
- `matches.csv` → match-level details (season, teams, venue, toss, winner, etc.)
- `deliveries.csv` → ball-by-ball (not used here).

The training script automatically downloads and uses `matches.csv`.

