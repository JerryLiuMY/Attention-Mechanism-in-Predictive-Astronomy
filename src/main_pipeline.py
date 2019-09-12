import numpy as np
import os
import json
import pickle
import matplotlib.pyplot as plt
from global_setting import DATA_FOLDER
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from model.standard_lstm import StandardLSTM
np.random.seed(1)


class main_pipelien:
    # Note: This need to be run in python2
    def __init__(self, crts_id):
        # Configuration
        self.crts_id = crts_id
        self.data_config = json.load(open('./config/data_config.json'))
        self.model_config = json.load(open('./config/model_config.json'))

        # Individual data
        self.basic_data_load_path = os.path.join(DATA_FOLDER, 'processed_data', 'basic', str(crts_id) + '.pickle')
        with open(self.basic_data_load_path, 'rb') as handle:
            data_dict = pickle.load(handle)
        self.mag_list_train = data_dict['mag_list_train']
        self.mag_list_cross = data_dict['mag_list_cross']
        self.mag_list_test = data_dict['mag_list_test']
        self.magerr_list_train = data_dict['magerr_list_train']
        self.magerr_list_cross = data_dict['magerr_list_cross']
        self.magerr_list_test = data_dict['magerr_list_test']
        self.t_list_train = data_dict['t_list_train']
        self.t_list_cross = data_dict['t_list_cross']
        self.t_list_test = data_dict['t_list_test']

        # All data
        self.basic_all_data_load_path = os.path.join(DATA_FOLDER, 'processed_data', 'basic', 'all.pickle')
        with open(self.basic_all_data_load_path, 'rb') as handle:
            all_data_dict = pickle.load(handle)
        self.mag_lists_train = all_data_dict['mag_list_train']
        self.mag_lists_cross = all_data_dict['mag_list_cross']
        self.mag_lists_test = all_data_dict['mag_list_test']
        self.magerr_lists_train = all_data_dict['magerr_list_train']
        self.magerr_lists_cross = all_data_dict['magerr_list_cross']
        self.magerr_lists_test = all_data_dict['magerr_list_test']
        self.t_lists_train = all_data_dict['t_list_train']
        self.t_lists_cross = all_data_dict['t_list_cross']
        self.t_lists_test = all_data_dict['t_list_test']

    def 

