import math
import numpy as np
import pandas as pd
import os
import joblib
import pickle
from global_setting import raw_data_folder, basic_data_folder, lstm_data_folder
from global_setting import WINDOW_LEN, TRAIN_RATIO, CROSS_RATIO
from sklearn.preprocessing import MinMaxScaler
np.random.seed(1)


class BasicDataProcessor:
    def __init__(self, crts_id):
        self.crts_id = crts_id
        self.load_data_folder()

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
        p1 = int(math.floor(length * TRAIN_RATIO))
        p2 = int(math.floor(length * (TRAIN_RATIO + CROSS_RATIO)))

        index_list = list(range(length))
        index_train = np.array(index_list[0:p1])
        index_cross = np.array(index_list[p1:p2])
        index_test = np.array(index_list[p2:])

        return index_train, index_cross, index_test


class LSTMDataProcessor:
    def __init__(self, crts_id):
        self.crts_id = crts_id
        self.load_data_folder()

    def load_data_folder(self):
        self.raw_data_folder = raw_data_folder
        self.basic_data_folder = basic_data_folder
        self.lstm_data_folder = lstm_data_folder

    def prepare_lstm_data(self):
        with open(os.path.join(self.basic_data_folder, str(self.crts_id) + '.pkl'), 'rb') as handle:
            data_dict = pickle.load(handle)

            mag_train = data_dict['mag_train'].reshape(-1, 1)
            mag_cross = data_dict['mag_cross'].reshape(-1, 1)
            mag_test = data_dict['mag_test'].reshape(-1, 1)

            delta_t_train = np.diff(data_dict['t_train']).reshape(-1, 1)
            delta_t_cross = np.diff(data_dict['t_cross']).reshape(-1, 1)
            delta_t_test = np.diff(data_dict['t_test']).reshape(-1, 1)

        scaled_mag_train, scaled_mag_cross, scaled_mag_test, mag_scaler = self.split_rescale(mag_train, mag_cross, mag_test, 'mag')
        scaled_delta_t_train, scaled_delta_t_cross, scaled_delta_t_test, delta_t_scaler = self.split_rescale(delta_t_train, delta_t_cross, delta_t_test, 'delta_t')

        X_train, y_train = self.create_X_y(scaled_mag_train, scaled_delta_t_train, WINDOW_LEN)
        X_cross, y_cross = self.create_X_y(scaled_mag_cross, scaled_delta_t_cross, WINDOW_LEN)
        X_test, y_test = self.create_X_y(scaled_mag_test, scaled_delta_t_test, WINDOW_LEN)

        X_y_data_dict = {'train_X': X_train, 'train_y': y_train, 'cross_X': X_cross, 'cross_y': y_cross,
                         'test_X': X_test, 'test_y': y_test}

        # Save X_y Data
        X_y_name = '_'.join([str(self.crts_id), 'X_y', 'window_len', str(WINDOW_LEN) + '.plk'])
        with open(os.path.join(self.lstm_data_folder, X_y_name), 'wb') as handle:
            pickle.dump(X_y_data_dict, handle, protocol=2)

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
        scaler_name = '_'.join([str(self.crts_id), data_type, 'scaler.pkl'])

        with open(os.path.join(self.lstm_data_folder, scaler_name), 'wb') as handle:
            joblib.dump(scaler, handle, protocol=2)

        return scaled_train, scaled_cross, scaled_test, scaler

    @staticmethod
    def create_X_y(scaled_mag, scaled_delta_t, window_len):
        X, y = [], []
        for i in range(0, np.shape(scaled_mag)[0] - window_len):
            features = []
            for j in range(i, i + window_len):
                feature = np.concatenate((scaled_mag[j], scaled_delta_t[j]), axis=0)
                features.append(feature)
            X.append(features)
            y.append(scaled_mag[i + window_len])

        X = np.array(X)
        y = np.array(y)

        return X, y


if __name__ == 'main':
    basic_data_processor = BasicDataProcessor(1001115026824)
    basic_data_processor.prepare_basic_data()
    lstm_data_processor = LSTMDataProcessor(1001115026824)
    lstm_data_processor.prepare_lstm_data()

