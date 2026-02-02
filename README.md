# 🏈 NFL Play Predictor

A machine learning application that predicts the next offensive play type (Rush, Pass, Special Teams) based on live game context.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Jupyter](https://img.shields.io/badge/Interface-Interactive%20Widgets-F37626)

## 📋 Project Overview

This project utilizes historical NFL play-by-play data (2013–2023) to train a Gradient Boosted Decision Tree (XGBoost) classifier. It moves beyond simple linear baselines to capture the non-linear, contextual nature of football strategy (e.g., how "4th & 1" differs at the goal line vs. midfield).

**Key Capabilities:**
* **Predictive Engine:** Classifies plays into `Rush`, `Pass`, `Special`, or `No Play` with high fidelity.
* **Interactive Dashboard:** A Jupyter-based GUI for real-time inference during live games.
* **Modular Architecture:** Separation of concerns between data processing (`src`), experimentation (`notebooks`), and production logic.

# **Getting Started**
**1. Clone the repo**
git clone [https://github.com/yourusername/nfl-play-predictor.git](https://github.com/yourusername/nfl-play-predictor.git)
cd nfl-play-predictor

**2. Create venv**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

**3. Install dependencies**
pip install -r requirements.txt

# **Usage**
1. Train the Model
Run notebooks/01_model_training.ipynb to process the raw data and generate the model artifact.
- Input: Raw NFL Play-by-Play CSV (placed in data/raw/)
- Output: data/models/nfl_sgd_pipeline.joblib
- Metric: Classification Report (Precision/Recall/F1)

2. Live Inference Dashboard
- Run notebooks/02_interactive_inference.ipynb to launch the GUI.
- Select the current game state (Down, Distance, Field Position, Teams).
- Click Predict Next Play to see the model's confidence scores.

🧠 Model Details
- Algorithm: XGBoost Classifier (Gradient Boosted Trees)
- Objective: Multi-class Softprob
-Features:
-- Context: Quarter, MinutesRemaining, SecondsRemaining
-- State: Down, ToGo (Yards needed), YardLine (Field position)
-- Teams: OffenseID, DefenseID (Categorical Embeddings)
-- Performance: ~75% Accuracy (significantly outperforming the ~50% linear baseline).

🔮 Future Improvements
-Feature Engineering: Integrate "Score Differential" (a critical predictor for run/pass splits).
-Time Series: Combine Minutes/Seconds into a continuous SecondsRemaining feature.
- Deployment: Wrap the inference engine in a Flask/Streamlit app for web access.

Created using NFL Savant's play-by-play data. All libraries, APIs, and data references should be considered attributed to their respective owners.

Any references to this repository, code, or linked code should be attributed to Schraeder Technologies.

## 🏗️ Architecture

The repository follows a production-grade Data Science structure:

```text
nfl-play-predictor/
├── data/
│   ├── raw/                   # Historical CSV data
│   └── models/                # Serialized XGBoost pipelines (.joblib)
├── notebooks/
│   ├── 01_model_training.ipynb       # ETL, Encoding, Training, & Evaluation
│   └── 02_interactive_inference.ipynb # User-facing Prediction Dashboard
├── src/
│   ├── __init__.py
│   └── nfl_utils.py           # Shared logic for cleaning & feature engineering
├── requirements.txt           # Dependency management
└── README.md
