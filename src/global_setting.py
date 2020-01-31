import os
import pandas as pd
import json
import fnmatch
DATA_FOLDER = '/Users/mingyu/Desktop/dataset'  # '/Users/mingyu/Desktop/dataset'

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
basic_data_folder = os.path.join(DATA_FOLDER, 'processed_data', 'basic')
lstm_data_folder = os.path.join(DATA_FOLDER, 'processed_data', 'lstm')

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


# self.prepare_basic_data()
# inputs: len(content['Mag']) = (num_data, )
# inputs: len(content['Magerr']) = (num_data, )
# inputs: len(content['MJD']) = (num_data, )
# output: shape(mag_list_train) = (num_train_data, )
# output: shape(mag_cross_train) = (num_cross_data, )
# output: shape(mag_test_train) = (num_test_data, )

# self.prepare_rescale_mag()
# input: shape(mag_list_train) = (num_train_data, )
# input: shape(mag_list_cross) = (num_cross_data, )
# input: shape(mag_list_test) = (num_test_data, )
# output: shape(scaled_mag_list_train) = (num_train_data - 1, 1)
# output: shape(scaled_mag_list_cross) = (num_cross_data - 1, 1)
# output: shape(scaled_mag_list_test) = (num_test_data - 1, 1)

# self.create_X_y()
# input: shape(scaled_mag_list) = (num_data - 1, 1)
# input: shape(scaled_delta_t_list) = (num_data - 1, 1)
# output: shape(X) = (num_data - window_len - 2, window_len, 2)
# output: shape(y) = (num_data - window_len - 2, 1)

# self.prepare_lstm_data()
# input: scaled_mag_list & scaled_delta_t_list
# output: shape(X_train) = (num_train_data - window_len - 2, window_len, 2)
# output: shale(y_train) = (num_train_data - window_len - 2, 1)
# output: shape(X_cross) = (num_cross_data - window_len - 2, window_len, 2)
# output: shale(y_cross) = (num_cross_data - window_len - 2, 1)
# output: shape(X_test) = (num_test_data - window_len - 2, window_len, 2)
# output: shale(y_test) = (num_test_data - window_len - 2, 1)