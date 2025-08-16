import streamlit as st
import pandas as pd
import joblib
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.txt")

st.set_page_config(page_title=" IPL Winner Predictor", page_icon="", layout="centered")

# ------------------- CUSTOM CSS -------------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }
    .stSelectbox label, .stRadio label {
        font-weight: bold;
        color: #f5f5f5 !important;
    }
    .stButton button {
        background-color: #ff5722 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1.2em;
    }
    .stButton button:hover {
        background-color: #e64a19 !important;
    }
    .prediction-card {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(" IPL Winner Predictor")

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        cols = f.read().split(",")
    return model, cols

@st.cache_data
def load_reference_data():
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "patrickb1912/ipl-complete-dataset-20082020",
        "matches.csv"
    )
    venues = sorted(df["venue"].dropna().unique().tolist())
    seasons = sorted(df["season"].dropna().unique().tolist())

    #  Current IPL teams (2025, 10 teams)
    present_teams = [
        "Chennai Super Kings",
        "Mumbai Indians",
        "Royal Challengers Bangalore",
        "Kolkata Knight Riders",
        "Sunrisers Hyderabad",
        "Rajasthan Royals",
        "Delhi Capitals",
        "Punjab Kings",
        "Gujarat Titans",
        "Lucknow Super Giants"
    ]
    return present_teams, venues, seasons

model, feature_cols = load_model()
teams, venues, seasons = load_reference_data()

# ------------------- INPUT FORM -------------------
st.markdown("### ⚡ Enter Match Details")

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    team_a = col1.selectbox("Team A (team1)", options=teams)
    team_b = col2.selectbox("Team B (team2)", options=teams)
    venue = st.selectbox(" Venue", options=venues)
    season = st.selectbox(" Season", options=seasons)
    toss_winner = st.selectbox(" Toss Winner", options=[team_a, team_b])
    toss_decision = st.radio("Toss Decision", options=["bat","field"], horizontal=True)
    submitted = st.form_submit_button(" Predict Winner")

# ------------------- PREDICTION -------------------
if submitted:
    if team_a == team_b:
        st.error(" Team A and Team B must be different.")
    else:
        X = pd.DataFrame([{
            "season": season,
            "venue": venue,
            "team1": team_a,
            "team2": team_b,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
        }])
        try:
            proba_a = float(model.predict_proba(X)[:,1][0])
            proba_b = 1.0 - proba_a

            st.markdown("##  Prediction Results")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <h3>{team_a}</h3>
                        <h2 style="color:#4caf50;">{proba_a*100:.1f}%</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <h3>{team_b}</h3>
                        <h2 style="color:#f44336;">{proba_b*100:.1f}%</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.info(" Model trained on 2008-2024 IPL data. Retrain with updated data for best accuracy.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
