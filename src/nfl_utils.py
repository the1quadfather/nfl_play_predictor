import pandas as pd
import numpy as np

# Comprehensive mapping merging historical (SD, STL) and current teams
# ID 33 was identified in the dataset as 'Unknown' and is excluded from this valid map
TEAM_MAPPING = {
    "ARI": 1, "ATL": 2, "BAL": 3, "BUF": 4, "CAR": 5, "CHI": 6,
    "CIN": 7, "CLE": 8, "DAL": 9, "DEN": 10, "DET": 11, "GB": 12,
    "HOU": 13, "IND": 14, "JAX": 15, "KC": 16, "LV": 17, "OAK": 17,
    "LAC": 18, "SD": 18, "LA": 19, "STL": 19, "MIA": 20, "MIN": 21,
    "NE": 22, "NO": 23, "NYG": 24, "NYJ": 25, "PHI": 26, "PIT": 27,
    "SF": 28, "SEA": 29, "TB": 30, "TEN": 31, "WAS": 32
}

PLAY_TYPE_MAPPING = {
    1: "Rush",
    2: "Pass",
    3: "Special",
    4: "No Play"
}

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads data and performs production-grade cleaning.
    """
    print(f"Loading data from {file_path}...")
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file at {file_path}. Check your data/raw folder.")

    # 1. Standardize Columns
    required_cols = [
        "SeasonYear", "Quarter", "Minute", "Second", 
        "Down", "ToGo", "YardLine", "OFF", "DEF", "Rush/Pass/Special"
    ]
    
    # 2. Drop Target NaNs (We cannot train without a label)
    df_clean = df.dropna(subset=["Rush/Pass/Special"]).copy()
    
    # 3. Clean Team IDs
    # The dataset contains IDs (floats) in 'OFF'/'DEF'. We cast to int.
    # We also filter out IDs not in our known map (e.g., ID 33) to prevent errors.
    valid_ids = set(TEAM_MAPPING.values())
    
    df_clean = df_clean[
        df_clean['OFF'].isin(valid_ids) & 
        df_clean['DEF'].isin(valid_ids)
    ]
    
    df_clean['OFF'] = df_clean['OFF'].astype(int)
    df_clean['DEF'] = df_clean['DEF'].astype(int)
    
    print(f"Data cleaned. Rows remaining: {len(df_clean)}")
    return df_clean

def get_features_and_target(df: pd.DataFrame):
    """
    Splits the dataframe into X (features) and y (target).
    """
    features = [
        "SeasonYear", "Quarter", "Minute", "Second", 
        "Down", "ToGo", "YardLine", "OFF", "DEF"
    ]
    
    X = df[features]
    y = df["Rush/Pass/Special"]
    
    return X, y