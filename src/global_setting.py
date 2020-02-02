import os
import json
import fnmatch
DATA_FOLDER = '/Users/mingyu/Desktop/dataset'
WINDOW_LEN = 10
N_WALKERS = 1000
TRAIN_RATIO = 0.6
CROSS_RATIO = 0.2
TEST_RATIO = 0.2

# ----------------------------------- List of Lightcurves -----------------------------------
lightcurve_list = []
for file in os.listdir(os.path.join(DATA_FOLDER, 'raw_data')):
    if fnmatch.fnmatch(file, '*.csv'):
        lightcurve = file.split('.')[0]
        lightcurve_list.append(lightcurve)

# ----------------------------------- Configuration -----------------------------------
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


# input: shape(t_train) = (num_data, )
# input: shape(mag_train) = (num_train_data, )
# input: shape(magerr_train) = (num_train_data, )
# shape(X_train) = (num_train_data - window_len, window_len, 2)
# shale(y_train) = (num_train_data - window_len, 1)


# window_len = 3
# 1(1) 2(2) 3(3) 4(4) 5(5) 6(6) 7(7)
# 1-2(1) 2-3(2) 3-4(3) 4-5(4) 5-6(5) 6-7(6)

# X
# 1-2(1) 2-3(2) 3-4(3)  --> 4
# 2-3(2) 3-4(3) 4-5(4)  --> 5
# 3-4(3) 4-5(4) 5-6(5)  --> 6
# 4-5(4) 5-6(5) 6-7(6)  --> 7

# DISCRETE t_pred
# 3 4 5 6 7
# DISCRETE
# 1-2(1) 2-3(2) 3-4(3)  --> 4
# 2-3(2) 3-4(3) 4-5(4)  --> 5
# 3-4(3) 4-5(4) 5-6(5)  --> 6
# 4-5(4) 5-6(5) 6-7(6)  --> 7

# CONTINUOUS t_pred
# 3.0 3.1 3.2 3.3 3.4
# CONTINUOUS
# 1-2(1)       2-3(2)       3.0-3.1(3)    --> 3.1
# 2-3(2)       3.0-3.1(3)   3.1-3.2(3.1)  --> 3.2
# 3.0-3.1(3)   3.1-3.2(3.1) 3.2-3.3(3.2)  --> 3.3
# 3.1-3.2(3.1) 3.2-3.3(3.2) 3.3-3.4(3.3)  --> 3.4
