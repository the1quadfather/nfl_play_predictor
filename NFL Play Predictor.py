# NFL Playcall Predictor
# Based on down, yard line, yards-to-go, time, quarter, score differential, predict whether the next play is
# a run or a pass
# Play-by-play data: 2013-2023, NFLSavant.com


import pandas as pd
import numpy as np


# PBP CSV data import
df_x = pd.read_csv('NFL_PBP_combined_playground_csv.csv', usecols=["SeasonYear", "Quarter", "Minute",
                                                                   "Second", "Down", "ToGo", "YardLine", "OFF", "DEF"])
df_y = pd.read_csv('NFL_PBP_combined_playground_csv.csv', usecols=["RushPassSpecial"])

# Perform multi-class classification using SVM (SVC)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
X = df_x
Y = np.ravel(df_y)
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, Y)
Pipeline(steps=[('scaler', StandardScaler()), 'svc', SVC(gamma='auto')])