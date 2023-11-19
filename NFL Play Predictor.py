# NFL Playcall Predictor
# Based on down, yard line, yards-to-go, time, quarter, score differential, predict whether the next play is
# a run or a pass
# Play-by-play data: 2013-2023, NFLSavant.com

import pandas as pd
import numpy as np
from tensorflow import keras
from keras import layers
import os

#