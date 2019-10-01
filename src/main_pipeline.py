import numpy as np
import os
import json
import pickle
import joblib
from global_setting import DATA_FOLDER
from global_setting import raw_data_path, basic_data_path, basic_lstm_data_path
from utils.data_processor import BasicDataProcessor, LSTMDataProcessor
from model.basic_lstm import BasicLSTM
from model.test_lstm import TestLSTM


class MainPipeline:
    def __init__(self, crts_id):
        # Configuration
        self.crts_id = crts_id
        self.laod_config()
        self.load_crts_list()
        # self.prepare_all_data()

        # Data Path
        self.raw_data_path = raw_data_path
        self.basic_data_path = basic_data_path
        self.basic_lstm_data_path = basic_lstm_data_path

        # Basic Data Name
        self.basic_data_name = str(self.crts_id) + '.pkl'

        # LSTM Data Name
        self.rescaled_mag_name = str(self.crts_id) + '_rescaled_mag' + '.pkl'
        self.mag_scaler_name = str(self.crts_id) + '_mag_scaler' + '.pkl'
        self.rescaled_delta_t_name = str(self.crts_id) + '_rescaled_delta_t' + '.pkl'
        self.delta_t_scaler_name = str(self.crts_id) + '_delta_t_scaler' + '.pkl'
        self.standard_X_y_name = str(self.crts_id) + '_X_y' + '_window_len_' + str(self.basic_lstm_window_len) + '.plk'

    # ----------------------------------- Load Configuration -----------------------------------
    def laod_config(self):
        self.data_config = json.load(open('./config/data_config.json'))
        self.model_config = json.load(open('./config/model_config.json'))
        self.basic_lstm_window_len = self.model_config["basic_lstm"]["window_len"]

    def load_crts_list(self):
        crts_list = []
        for file in os.listdir(os.path.join(DATA_FOLDER, 'raw_data')):
            if file.endswith(".csv"):
                crts_list.append(file.split('.')[0])
        self.crts_list = crts_list

    # ----------------------------------- Load Data -----------------------------------
    def prepare_all_data(self):
        for crts_id in self.crts_list:
            # Basic Data
            basic_data_processor = BasicDataProcessor(crts_id)
            basic_data_processor.prepare_basic_data()

            # LSTM Data
            lstm_data_processor = LSTMDataProcessor(crts_id)
            lstm_data_processor.prepare_rescale_mag()
            lstm_data_processor.prepare_rescale_delta_t()
            lstm_data_processor.prepare_basic_lstm_data()

    def prepare_individual_data(self):
        # Basic Data
        basic_data_processor = BasicDataProcessor(self.crts_id)
        basic_data_processor.prepare_basic_data()

        # LSTM Data
        lstm_data_processor = LSTMDataProcessor(self.crts_id)
        lstm_data_processor.prepare_rescale_mag()
        lstm_data_processor.prepare_rescale_delta_t()
        lstm_data_processor.prepare_basic_lstm_data()

    def load_individual_data(self):
        # Individual data
        self.basic_data_load_path = os.path.join(DATA_FOLDER, 'processed_data', 'basic', str(self.crts_id) + '.pkl')
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

    # ----------------------------------- Run Single Model -----------------------------------
    def run_standard_lstm(self, implementation='multi'):
        standard_lstm = BasicLSTM(self.crts_id, phased=False)
        standard_lstm.build_model()

        mag_scaler = joblib.load(os.path.join(self.basic_lstm_data_path, self.mag_scaler_name))
        with open(os.path.join(self.basic_lstm_data_path, self.standard_X_y_name), 'rb') as handle:
            X_y_data_dict = pickle.load(handle)
            train_X = X_y_data_dict['train_X']
            train_y = X_y_data_dict['train_y']
            cross_X = X_y_data_dict['cross_X']
            cross_y = X_y_data_dict['cross_y']
            test_X = X_y_data_dict['test_X']
            test_y = X_y_data_dict['test_y']

        standard_lstm.fit_model(train_X, train_y, cross_X, cross_y, test_X, test_y)
        if implementation == 'multi':
            y_inter, cross_y_pred, test_y_pred = standard_lstm.multiple_step_prediction(train_X, cross_X, test_X, mag_scaler)
        else:
            y_inter, cross_y_pred, test_y_pred = standard_lstm.one_step_prediction(train_X, cross_X, test_X, mag_scaler)

        standard_lstm.plot_prediction(self.mag_list_train, self.mag_list_cross, self.mag_list_test, self.t_list_cross,
                                      self.t_list_train, self.t_list_test, self.magerr_list_train,
                                      self.magerr_list_cross, self.magerr_list_test, y_inter, cross_y_pred, test_y_pred)

    def run_phased_lstm(self, implementation='multi'):
        phased_lstm = BasicLSTM(self.crts_id, phased=True)
        phased_lstm.build_model()

        mag_scaler = joblib.load(os.path.join(self.basic_lstm_data_path, self.mag_scaler_name))
        with open(os.path.join(self.basic_lstm_data_path, self.standard_X_y_name), 'rb') as handle:
            X_y_data_dict = pickle.load(handle)
            train_X = X_y_data_dict['train_X']
            train_y = X_y_data_dict['train_y']
            cross_X = X_y_data_dict['cross_X']
            cross_y = X_y_data_dict['cross_y']
            test_X = X_y_data_dict['test_X']
            test_y = X_y_data_dict['test_y']

        phased_lstm.fit_model(train_X, train_y, cross_X, cross_y, test_X, test_y)
        if implementation == 'multi':
            y_inter, cross_y_pred, test_y_pred = phased_lstm.multiple_step_prediction(train_X, cross_X, test_X, mag_scaler)
        else:
            y_inter, cross_y_pred, test_y_pred = phased_lstm.one_step_prediction(train_X, cross_X, test_X, mag_scaler)

        phased_lstm.plot_prediction(self.mag_list_train, self.mag_list_cross, self.mag_list_test, self.t_list_cross,
                                    self.t_list_train, self.t_list_test, self.magerr_list_train,
                                    self.magerr_list_cross, self.magerr_list_test, y_inter, cross_y_pred, test_y_pred)

    def run_test_lstm(self, implementation='multi'):
        phased_lstm = TestLSTM(self.crts_id, phased=True)
        phased_lstm.build_model()

        mag_scaler = joblib.load(os.path.join(self.basic_lstm_data_path, self.mag_scaler_name))
        with open(os.path.join(self.basic_lstm_data_path, self.standard_X_y_name), 'rb') as handle:
            X_y_data_dict = pickle.load(handle)
            train_X = X_y_data_dict['train_X']
            train_y = X_y_data_dict['train_y']
            cross_X = X_y_data_dict['cross_X']
            cross_y = X_y_data_dict['cross_y']
            test_X = X_y_data_dict['test_X']
            test_y = X_y_data_dict['test_y']

        phased_lstm.fit_model(train_X, train_y, cross_X, cross_y, test_X, test_y)
        if implementation == 'multi':
            y_inter, cross_y_pred, test_y_pred = phased_lstm.multiple_step_prediction(train_X, cross_X, test_X, mag_scaler)
        else:
            y_inter, cross_y_pred, test_y_pred = phased_lstm.one_step_prediction(train_X, cross_X, test_X, mag_scaler)

        phased_lstm.plot_prediction(self.mag_list_train, self.mag_list_cross, self.mag_list_test, self.t_list_cross,
                                    self.t_list_train, self.t_list_test, self.magerr_list_train,
                                    self.magerr_list_cross, self.magerr_list_test, y_inter, cross_y_pred, test_y_pred)

    # ----------------------------------- Run Hybrid Model -----------------------------------

if __name__ == 'main':
    instance = MainPipeline(1001115026824)
    instance.prepare_all_data()
    instance.prepare_individual_data()
    instance.load_individual_data()

    instance.run_phased_lstm()


