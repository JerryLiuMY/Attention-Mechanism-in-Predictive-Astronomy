import numpy as np
import os
import json
import pickle
import joblib
from global_setting import DATA_FOLDER
from utils.data_processor import BasicDataProcessor, LSTMDataProcessor
from model.standard_lstm import StandardLSTM
np.random.seed(1)


class MainPipeline:
    def __init__(self, crts_id):
        # Configuration
        self.crts_id = crts_id
        self.laod_config()
        self.prepare_data()
        self.load_individual_data()

        # Path
        self.standard_lstm_data_path = os.path.join(DATA_FOLDER, 'processed_data', 'standard_lstm')

    # ----------------------------------- Load Configuration -----------------------------------
    def laod_config(self):
        self.data_config = json.load(open('./config/data_config.json'))
        self.model_config = json.load(open('./config/model_config.json'))
        self.windoe_len = self.model_config["standard_lstm"]["window_len"]

    # ----------------------------------- Load Data -----------------------------------
    def prepare_data(self):
        # Basic Data
        basic_data_processor = BasicDataProcessor()
        basic_data_processor.prepare_basic_data()

        # LSTM Data
        lstm_data_processor = LSTMDataProcessor()
        lstm_data_processor.prepare_rescale_mag()
        lstm_data_processor.prepare_rescale_delta_t()
        lstm_data_processor.prepare_standard_lstm_data()

    def load_individual_data(self):
        # Individual data
        self.basic_data_load_path = os.path.join(DATA_FOLDER, 'processed_data', 'basic', str(self.crts_id) + '.pickle')
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

    # ----------------------------------- Run Model -----------------------------------
    def run_standard_lstm(self):
        standard_lstm = StandardLSTM(self.crts_id)
        standard_lstm.build_model()

        mag_scaler = joblib.load(self.standard_lstm_data_path, 'mag_scaler_' + str(self.crts_id) + '.pkl')
        X_y_file_name = 'X_y_' + str(self.crts_id) + '_window_len_' + str(self.windoe_len) + '.plk'
        with open(os.path.join(self.standard_lstm_data_path, X_y_file_name), 'wb') as handle:
            X_y_data_dict = pickle.load(handle)
            train_X = X_y_data_dict['train_X']
            train_y = X_y_data_dict['train_y']
            cross_X = X_y_data_dict['cross_X']
            cross_y = X_y_data_dict['cross_y']
            test_X = X_y_data_dict['test_X']
            test_y = X_y_data_dict['test_y']

        with open(os.path.join(self.standard_lstm_data_path, 'rescaled_delta_t_' + str(self.crts_id) + '.pkl'), 'wb') as handle:
            scaled_delta_t_data_dict = pickle.load(handle)
            scaled_delta_t_list_train = scaled_delta_t_data_dict['scaled_delta_t_list_train']
            scaled_delta_t_list_cross = scaled_delta_t_data_dict['scaled_delta_t_list_cross']
            scaled_delta_t_list_test = scaled_delta_t_data_dict['scaled_delta_t_list_test']

        standard_lstm.fit_model(train_X, train_y, cross_X, cross_y, test_X, test_y)
        # y_inter, cross_y_pred, test_y_pred = standard_lstm.one_step_prediction(self, train_X, cross_X, test_X, mag_scaler)
        y_inter, cross_y_pred, test_y_pred = standard_lstm.multiple_step_prediction(train_X, cross_X, test_X,
                                                                                    scaled_delta_t_list_cross,
                                                                                    scaled_delta_t_list_test,
                                                                                    mag_scaler)
        standard_lstm.plot_prediction(self.mag_list_train, self.mag_list_cross, self.mag_list_test, self.t_list_cross,
                                      self.t_list_train, self.t_list_test, self.magerr_list_train,
                                      self.magerr_list_cross, self.magerr_list_test, y_inter, cross_y_pred, test_y_pred)

    def run_phased_lstm(self):
        standard_lstm = StandardLSTM(self.crts_id)
        standard_lstm.build_model()

        mag_scaler = joblib.load(self.standard_lstm_data_path, 'mag_scaler_' + str(self.crts_id) + '.pkl')
        X_y_file_name = 'X_y_' + str(self.crts_id) + '_window_len_' + str(self.windoe_len) + '.plk'
        with open(os.path.join(self.standard_lstm_data_path, X_y_file_name), 'wb') as handle:
            X_y_data_dict = pickle.load(handle)
            train_X = X_y_data_dict['train_X']
            train_y = X_y_data_dict['train_y']
            cross_X = X_y_data_dict['cross_X']
            cross_y = X_y_data_dict['cross_y']
            test_X = X_y_data_dict['test_X']
            test_y = X_y_data_dict['test_y']

        with open(os.path.join(self.standard_lstm_data_path, 'rescaled_delta_t_' + str(self.crts_id) + '.pkl'), 'wb') as handle:
            scaled_delta_t_data_dict = pickle.load(handle)
            scaled_delta_t_list_train = scaled_delta_t_data_dict['scaled_delta_t_list_train']
            scaled_delta_t_list_cross = scaled_delta_t_data_dict['scaled_delta_t_list_cross']
            scaled_delta_t_list_test = scaled_delta_t_data_dict['scaled_delta_t_list_test']

        standard_lstm.fit_model(train_X, train_y, cross_X, cross_y, test_X, test_y)
        # y_inter, cross_y_pred, test_y_pred = standard_lstm.one_step_prediction(self, train_X, cross_X, test_X, mag_scaler)
        y_inter, cross_y_pred, test_y_pred = standard_lstm.multiple_step_prediction(train_X, cross_X, test_X,
                                                                                    scaled_delta_t_list_cross,
                                                                                    scaled_delta_t_list_test,
                                                                                    mag_scaler)
        standard_lstm.plot_prediction(self.mag_list_train, self.mag_list_cross, self.mag_list_test, self.t_list_cross,
                                      self.t_list_train, self.t_list_test, self.magerr_list_train,
                                      self.magerr_list_cross, self.magerr_list_test, y_inter, cross_y_pred, test_y_pred)

if __name__ == 'main':
    instance = MainPipeline(1001115026824)
    instance.run_standard_lstm()


