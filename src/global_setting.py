import os
import json
import numpy as np
import pandas as pd 

DATA_FOLDER = '/Users/mingyu/Desktop/dataset'

# ----------------------------------- Data Information -----------------------------------
# raw_data
raw_data_path = os.path.join(DATA_FOLDER, 'raw_data')
raw_data_name = 'str(crts_id).csv'

# basic_data
basic_data_path = os.path.join(DATA_FOLDER, 'processed_data', 'basic')
basic_data_name = 'str(crts_id).pkl'

# standard_lstm_data
basic_lstm_data_path = os.path.join(DATA_FOLDER, 'processed_data', 'basic_lstm')
rescaled_mag_name = 'str(crts_id)_rescaled_mag.pkl'
mag_scaler_name = 'str(crts_id)_mag_scaler.pkl'
rescaled_delta_t_name = 'str(crts_id)_rescaled_delta_t.pkl'
delta_t_scaler_name = 'str(crts_id)_delta_t_scaler.pkl'

# ----------------------------------- Model Information -----------------------------------
# standard_lstm_model
standard_lstm_model_folder = os.path.join(DATA_FOLDER, 'model', 'standard_lstm')

# phased_lstm_model
phased_lstm_model_folder = os.path.join(DATA_FOLDER, 'model', 'phased_lstm')

# ----------------------------------- Configuration Information -----------------------------------
data_config = json.load(open('./config/data_config.json'))
model_config = json.load(open('./config/model_config.json'))

