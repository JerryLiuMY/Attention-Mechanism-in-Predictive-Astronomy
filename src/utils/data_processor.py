import math
import numpy as np
import pandas as pd
import os
import json
import joblib
import pickle
from global_setting import raw_data_folder, basic_data_folder, lstm_data_folder, data_config, model_config
from sklearn.preprocessing import MinMaxScaler
np.random.seed(1)


class BasicDataProcessor:
    def __init__(self, crts_id):
        self.crts_id = crts_id
        self.load_data_config()
        self.load_data_folder()

    def load_data_config(self):
        self.train_ratio = data_config['data_loader']['train_ratio']
        self.cross_ratio = data_config['data_loader']['cross_ratio']

    def load_data_folder(self):
        self.raw_data_folder = raw_data_folder
        self.basic_data_folder = basic_data_folder

    def prepare_basic_data(self):
        with open(os.path.join(self.raw_data_folder, str(self.crts_id) + '.csv')) as handle:
            content = pd.read_csv(handle)
            mag_list_ = np.array(content['Mag'])
            magerr_list_ = np.array(content['Magerr'])
            mjd_list_ = np.array(content['MJD'])

            # Sort data
            index_list = np.argsort(mjd_list_)
            mag_list = mag_list_[index_list]
            magerr_list = magerr_list_[index_list]
            mjd_list = mjd_list_[index_list]
            t_list = mjd_list - mjd_list.min()

        # Save Basic Data
        index_train, index_cross, index_test = self.partition_index(len(mag_list))
        t_train, t_cross, t_test = t_list[index_train], t_list[index_cross], t_list[index_test]
        mag_train, mag_cross, mag_test = mag_list[index_train], mag_list[index_cross], mag_list[index_test]
        magerr_train, magerr_cross, magerr_test = magerr_list[index_train], magerr_list[index_cross], magerr_list[index_test]
        data_dict = {'t_train': t_train, 't_cross': t_cross, 't_test': t_test,
                     'mag_train': mag_train, 'mag_cross': mag_cross, 'mag_test': mag_test,
                     'magerr_train': magerr_train, 'magerr_cross': magerr_cross, 'magerr_test': magerr_test}

        with open(os.path.join(self.basic_data_folder, str(self.crts_id) + '.pkl'), 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=2)

    def partition_index(self, length):
        p1 = int(math.floor(length * self.train_ratio))
        p2 = int(math.floor(length * (self.train_ratio + self.cross_ratio)))

        index_list = list(range(length))
        index_train = np.array(index_list[0:p1])
        index_cross = np.array(index_list[p1:p2])
        index_test = np.array(index_list[p2:])

        return index_train, index_cross, index_test


class LSTMDataProcessor:
    def __init__(self, crts_id):
        self.crts_id = crts_id
        self.load_config()
        self.load_data_folder()

    def load_config(self):
        self.train_percent = data_config['data_loader']['train_ratio']
        self.cross_percent = data_config['data_loader']['cross_ratio']
        self.model_config = json.load(open('./config/model_config.json'))

    def load_data_folder(self):
        self.raw_data_folder = raw_data_folder
        self.basic_data_folder = basic_data_folder
        self.lstm_data_folder = lstm_data_folder

    def prepare_lstm_data(self):
        with open(os.path.join(self.basic_data_folder, str(self.crts_id) + '.pkl'), 'rb') as handle:
            data_dict = pickle.load(handle)

            mag_train = data_dict['mag_train'][1:].reshape(-1, 1)
            mag_cross = data_dict['mag_cross'][1:].reshape(-1, 1)
            mag_test = data_dict['mag_test'][1:].reshape(-1, 1)

            delta_t_train = self.delta_list(data_dict['t_train']).reshape(-1, 1)
            delta_t_cross = self.delta_list(data_dict['t_cross']).reshape(-1, 1)
            delta_t_test = self.delta_list(data_dict['t_test']).reshape(-1, 1)

        scaled_mag_train, scaled_mag_cross, scaled_mag_test, mag_scaler = self.split_rescale(mag_train, mag_cross, mag_test, 'mag')
        scaled_delta_t_train, scaled_delta_t_cross, scaled_delta_t_test, delta_t_scaler = self.split_rescale(delta_t_train, delta_t_cross, delta_t_test, 'delta_t')

        window_len = self.model_config["lstm"]["window_len"]
        X_train, y_train = self.create_X_y(scaled_mag_train, scaled_delta_t_train, window_len)
        X_cross, y_cross = self.create_X_y(scaled_mag_cross, scaled_delta_t_cross, window_len)
        X_test, y_test = self.create_X_y(scaled_mag_test, scaled_delta_t_test, window_len)

        X_y_data_dict = {'train_X': X_train, 'train_y': y_train, 'cross_X': X_cross, 'cross_y': y_cross,
                         'test_X': X_test, 'test_y': y_test}

        # Save X_y Data
        X_y_name = '_'.join([str(self.crts_id), 'X_y', 'window_len', str(window_len) + '.plk'])
        with open(os.path.join(self.lstm_data_folder, X_y_name), 'wb') as handle:
            pickle.dump(X_y_data_dict, handle, protocol=2)

    @staticmethod
    def delta_list(raw_list):
        delta_list = []
        for i in range(1, len(raw_list)):
            delta = raw_list[i] - raw_list[i - 1]
            delta_list.append(delta)
        delta_list = np.array(delta_list)

        return delta_list

    def split_rescale(self, train, cross, test, data_type):
        # Scale Individual Data
        full = np.concatenate((train, cross, test), axis=0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_full = scaler.fit_transform(full)

        train_len = np.shape(train)[0]
        cross_len = np.shape(cross)[0]
        scaled_train = scaled_full[:train_len]
        scaled_cross = scaled_full[train_len:(train_len + cross_len)]
        scaled_test = scaled_full[(train_len + cross_len):]
        train_name = '_'.join(['scaled', data_type, 'train'])
        cross_name = '_'.join(['scaled', data_type, 'cross'])
        test_name = '_'.join(['scaled', data_type, 'test'])
        scaled_data_dict = {train_name: scaled_train,
                            cross_name: scaled_cross,
                            test_name: scaled_test}

        rescaled_name = '_'.join([str(self.crts_id), 'rescaled', data_type + '.pkl'])
        scaler_name = '_'.join([str(self.crts_id), data_type, 'scaler.pkl'])
        with open(os.path.join(self.lstm_data_folder, rescaled_name), 'wb') as handle:
            pickle.dump(scaled_data_dict, handle, protocol=2)
        with open(os.path.join(self.lstm_data_folder, scaler_name), 'wb') as handle:
            joblib.dump(scaler, handle, protocol=2)

        return scaled_train, scaled_cross, scaled_test, scaler

    @staticmethod
    def create_X_y(scaled_mag_list, scaled_delta_t_list, window_len):
        X, y = [], []
        for i in range(1, np.shape(scaled_mag_list)[0] - window_len):
            features = []
            for j in range(i, i + window_len):
                feature = np.concatenate((scaled_mag_list[j], scaled_delta_t_list[j]), axis=0)
                features.append(feature)
            X.append(features)
            y.append(scaled_mag_list[i + window_len])

        X = np.array(X)
        y = np.array(y)

        return X, y


if __name__ == 'main':
    basic_data_processor = BasicDataProcessor(1001115026824)
    basic_data_processor.prepare_basic_data()

    lstm_data_processor = LSTMDataProcessor(1001115026824)
    lstm_data_processor.prepare_lstm_data()

