import os
import json
import numpy as np
import pandas as pd 

DATA_FOLDER = '/Users/mingyu/Desktop/dataset'

# Data Path
raw_data_path = os.path.join(DATA_FOLDER, 'raw_data')
basic_data_path = os.path.join(DATA_FOLDER, 'processed_data', 'basic')
standard_lstm_data_path = os.path.join(DATA_FOLDER, 'processed_data', 'standard_lstm')

# Configuration
data_config = json.load(open('./config/data_config.json'))
model_config = json.load(open('./config/model_config.json'))

# Data Description
# basic_data:
# standard_lstm:
# phased_lstm:
# carma: use basic data
# gp: use basic data
