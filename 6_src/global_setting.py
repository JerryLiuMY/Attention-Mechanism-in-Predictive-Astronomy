import os
import json
DATA_FOLDER = '/Users/mingyu/Desktop/dataset'


# ----------------------------------- Configuration -----------------------------------
data_config = json.load(open('./config/data_config.json'))
model_config = json.load(open('./config/model_config.json'))

# ----------------------------------- Data Path -----------------------------------
# raw_data
raw_data_folder = os.path.join(DATA_FOLDER, 'raw_data')
raw_data_name = 'str(crts_id).csv'

# basic_data
basic_data_folder = os.path.join(DATA_FOLDER, 'processed_data', 'basic')
basic_data_name = 'str(crts_id).pkl'

# carma_data
carma_data_folder = os.path.join(DATA_FOLDER, 'processed_data', 'basic')
carma_data_name = 'str(crts_id).pkl'

# vanilla_lstm_data
vanilla_lstm_data_folder = os.path.join(DATA_FOLDER, 'processed_data', 'vanilla_lstm')
rescaled_mag_name = 'str(crts_id)_rescaled_mag.pkl'
mag_scaler_name = 'str(crts_id)_mag_scaler.pkl'
rescaled_delta_t_name = 'str(crts_id)_rescaled_delta_t.pkl'
delta_t_scaler_name = 'str(crts_id)_delta_t_scaler.pkl'

# ----------------------------------- Model Path -----------------------------------
# carma model
carma_model_folder = os.path.join(DATA_FOLDER, 'model', 'carma')
carma_model_name = 'str(crts_id)_carma_model.pkl'

# vanilla_lstm model
vanilla_lstm_model_folder = os.path.join(DATA_FOLDER, 'model', 'vanilla_lstm')
vanilla_lstm_standard_model_name = 'str(crts_id)_vanilla_lstm_standard_model.pkl'
vanilla_lstm_phased_model_name = 'str(crts_id)_vanilla_lstm_phased_modell.pkl'

# attention_lstm_model
attention_lstm_model_folder = os.path.join(DATA_FOLDER, 'model', 'attention_lstm')

# ----------------------------------- Figure Path -----------------------------------
# carma figure
carma_figure_folder = os.path.join(DATA_FOLDER, 'figure', 'carma')
carma_average_figure_name = 'str(crts_id)_carma_average_figure.png'
carma_sample_figure_name = 'str(crts_id)_carma_sample_figure.png'

# vanilla_lstm figure
vanilla_lstm_figure_folder = os.path.join(DATA_FOLDER, 'figure', 'vanilla_lstm')
vanilla_lstm_standard_multi_figure_name = 'str(crts_id)_vanilla_lstm_standard_multi_figure.png'
vanilla_lstm_standard_single_figure_name = 'str(crts_id)_vanilla_lstm_standard_single_figure.png'
vanilla_lstm_phased_multi_figure_name = 'str(crts_id)_vanilla_lstm_phased_multi_figure.png'
vanilla_lstm_phased_single_figure_name = 'str(crts_id)_vanilla_lstm_phased_single_figure.png'

# attention_lstm_figure
attention_standard_lstm_figure_folder = os.path.join(DATA_FOLDER, 'figure', 'attention_lstm')

# ----------------------------------- Outline -----------------------------------
# CARIMA Process / OU Process
# Gaussian Process
# Standard LSTM
# Bayssian / MC Standard LSTM
# Attention Standard LSTM
# Phased LSTM
# Attention Phased LSTM
