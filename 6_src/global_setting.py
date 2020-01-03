import os
import pandas as pd
import json
import fnmatch
DATA_FOLDER = '/Users/mingyu/Desktop/dataset_2'  # '/Users/mingyu/Desktop/dataset'

# ----------------------------------- List of Lightcurves -----------------------------------
lightcurve_list = []
for file in os.listdir(os.path.join(DATA_FOLDER, 'raw_data')):
    if fnmatch.fnmatch(file, '*.csv'):
        lightcurve = file.split('.')[0]
        lightcurve_list.append(lightcurve)

# ----------------------------------- Configuration -----------------------------------
data_config = json.load(open('./config/data_config.json'))
model_config = json.load(open('./config/model_config.json'))

# ----------------------------------- Data Path -----------------------------------
# raw_data
raw_data_folder = os.path.join(DATA_FOLDER, 'raw_data')
# raw_data_name = 'str(crts_id).csv'


# basic_data
basic_data_folder = os.path.join(DATA_FOLDER, 'processed_data', 'basic')
# basic_data_name = 'str(crts_id).pkl'


# carma_data
carma_data_folder = os.path.join(DATA_FOLDER, 'processed_data', 'basic')
# carma_data_name = 'str(crts_id).pkl'


# vanilla_lstm_data
vanilla_lstm_data_folder = os.path.join(DATA_FOLDER, 'processed_data', 'vanilla_lstm')
# rescaled_mag_name = 'str(crts_id)_rescaled_mag.pkl'
# mag_scaler_name = 'str(crts_id)_mag_scaler.pkl'
# rescaled_delta_t_name = 'str(crts_id)_rescaled_delta_t.pkl'
# delta_t_scaler_name = 'str(crts_id)_delta_t_scaler.pkl'

# ----------------------------------- Model Path -----------------------------------
# gp model
gp_model_folder = os.path.join(DATA_FOLDER, 'model', 'gp')
# gp_model_name = 'str(crts_id)_gp_model.pkl'


# carma model
carma_model_folder = os.path.join(DATA_FOLDER, 'model', 'carma')
# carma_model_name = 'str(crts_id)_carma_model.pkl'


# vanilla_lstm model
vanilla_lstm_model_folder = os.path.join(DATA_FOLDER, 'model', 'vanilla_lstm')
# vanilla_lstm_standard_model_name = 'str(crts_id)_vanilla_lstm_standard_model.h5'
# vanilla_lstm_phased_model_name = 'str(crts_id)_vanilla_lstm_phased_model.h5'


# attention_lstm_model
attention_lstm_model_folder = os.path.join(DATA_FOLDER, 'model', 'attention_lstm')

# ----------------------------------- Figure Path -----------------------------------
# gp figure
gp_figure_folder = os.path.join(DATA_FOLDER, 'figure', 'gp')
# gp_single_figure_name = 'str(crts_id)_gp_single_figure.png'
# gp_multi_figure_name = 'str(crts_id)_gp_multi_figure.png'


# carma figure
carma_figure_folder = os.path.join(DATA_FOLDER, 'figure', 'carma')
# carma_average_figure_name = 'str(crts_id)_carma_average_figure.png'
# carma_sample_figure_name = 'str(crts_id)_carma_sample_figure.png'


# vanilla_lstm figure
vanilla_lstm_figure_folder = os.path.join(DATA_FOLDER, 'figure', 'vanilla_lstm')
# vanilla_lstm_standard_multi_figure_name = 'str(crts_id)_vanilla_lstm_standard_multi_figure.png'
# vanilla_lstm_standard_single_figure_name = 'str(crts_id)_vanilla_lstm_standard_single_figure.png'
# vanilla_lstm_phased_multi_figure_name = 'str(crts_id)_vanilla_lstm_phased_multi_figure.png'
# vanilla_lstm_phased_single_figure_name = 'str(crts_id)_vanilla_lstm_phased_single_figure.png'


# attention_lstm_figure
attention_lstm_figure_folder = os.path.join(DATA_FOLDER, 'figure', 'attention_lstm')
# attention_lstm_standard_multi_figure_name = 'str(crts_id)_attention_lstm_standard_multi_figure.png'
# attention_lstm_standard_single_figure_name = 'str(crts_id)_attentiona_lstm_standard_single_figure.png'
# attention_lstm_phased_multi_figure_name = 'str(crts_id)_attention_lstm_phased_multi_figure.png'
# attention_lstm_phased_single_figure_name = 'str(crts_id)_attention_lstm_phased_single_figure.png'


# ----------------------------------- Result Path -----------------------------------
# csv file
result_csv = os.path.join(DATA_FOLDER, 'result', 'result.csv')

# ----------------------------------- Outline -----------------------------------
# CARIMA Process / OU Process
# Gaussian Process
# Standard LSTM
# Bayssian / MC Standard LSTM
# Attention Standard LSTM
# Phased LSTM
# Attention Phased LSTM
