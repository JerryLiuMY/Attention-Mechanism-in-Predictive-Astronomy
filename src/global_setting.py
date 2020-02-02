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
crts_list = []
for file in os.listdir(os.path.join(DATA_FOLDER, 'raw_data')):
    if fnmatch.fnmatch(file, '*.csv'):
        lightcurve = file.split('.')[0]
        crts_list.append(lightcurve)

model_config = json.load(open('./config/model_config.json'))
raw_data_folder = os.path.join(DATA_FOLDER, 'raw_data')
basic_data_folder = os.path.join(DATA_FOLDER, 'processed_data', 'basic')
lstm_data_folder = os.path.join(DATA_FOLDER, 'processed_data', 'lstm')

# ----------------------------------- Model Path -----------------------------------
model_folder = os.path.join(DATA_FOLDER, 'model')
gp_figure_folder = os.path.join(DATA_FOLDER, 'figure')
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
