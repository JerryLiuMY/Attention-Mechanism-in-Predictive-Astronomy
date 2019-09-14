import math
import random
import numpy as np
import pandas as pd
import os
from global_setting import DATA_FOLDER
from sklearn.preprocessing import MinMaxScaler
import json
import joblib
import pickle
import glob, os
np.random.seed(1)


class DataProcessor:

    def __init__(self):
        # Configuration
        self.load_config()
        self.load_crts_list()

        # Paths
        self.raw_data_path = os.path.join(DATA_FOLDER, 'raw_data')
        self.basic_data_path = os.path.join(DATA_FOLDER, 'processed_data', 'basic')
        self.standard_lstm_data_path = os.path.join(DATA_FOLDER, 'processed_data', 'standard_lstm')

    def load_config(self):
        self.data_config = json.load(open('./config/data_config.json'))
        self.train_percent = self.data_config['data_loader']['train_partition']
        self.cross_percent = self.data_config['data_loader']['cross_partition']
        self.model_config = json.load(open('./config/model_config.json'))

    def load_crts_list(self):
        crts_list = []
        for file in os.listdir(os.path.join(DATA_FOLDER, 'raw_data')):
            if file.endswith(".csv"):
                crts_list.append(file.split('.')[0])
        self.crts_list = crts_list

    # ------------------------- Basic Data Processor -------------------------
    def partition_index(self, length):
        p1 = int(math.floor(length * self.train_percent))
        p2 = int(math.floor(length * (self.train_percent + self.cross_percent)))

        index_list = list(range(length))
        index_train = np.array(index_list[0:p1])
        index_cross = np.array(index_list[p1:p2])
        index_test = np.array(index_list[p2:])

        return index_train, index_cross, index_test

    def prepare_basic_data(self):
        for crts_id in self.crts_list:
            with open(os.path.join(self.raw_data_path, str(crts_id) + '.csv')) as handle:
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

            # Save Individual Data
            index_train, index_cross, index_test = self.partition_index(len(mag_list))
            mag_list_train = mag_list[index_train]
            mag_list_cross = mag_list[index_cross]
            mag_list_test = mag_list[index_test]
            magerr_list_train = magerr_list[index_train]
            magerr_list_cross = magerr_list[index_cross]
            magerr_list_test = magerr_list[index_test]
            t_list_train = t_list[index_train]
            t_list_cross = t_list[index_cross]
            t_list_test = t_list[index_test]

            data_dict = {'mag_list_train': np.expand_dims(mag_list_train, axis=0),
                         'mag_list_cross': np.expand_dims(mag_list_cross, axis=0),
                         'mag_list_test': np.expand_dims(mag_list_test, axis=0),
                         'magerr_list_train': np.expand_dims(magerr_list_train, axis=0),
                         'magerr_list_cross': np.expand_dims(magerr_list_cross, axis=0),
                         'magerr_list_test': np.expand_dims(magerr_list_test, axis=0),
                         't_list_train': np.expand_dims(t_list_train, axis=0),
                         't_list_cross': np.expand_dims(t_list_cross, axis=0),
                         't_list_test': np.expand_dims(t_list_test, axis=0)}

            with open(os.path.join(self.basic_data_path, str(crts_id) + '.pkl'), 'wb') as handle:
                pickle.dump(data_dict, handle)

    def prepare_basic_dataset(self):
        mag_lists_train, mag_lists_cross, mag_lists_test = [], [], []
        magerr_lists_train, magerr_lists_cross, magerr_lists_test = [], [], []
        t_lists_train, t_lists_cross, t_lists_test = [], [], []
        for crts_id in self.crts_list:
            with open(os.path.join(self.basic_data_path, str(crts_id) + '.pkl'), 'rb') as handle:
                data_dict = pickle.load(handle)

                # Retrieve Individual Data
                mag_list_train = data_dict['mag_list_train']
                mag_list_cross = data_dict['mag_list_cross']
                mag_list_test = data_dict['mag_list_test']
                magerr_list_train = data_dict['magerr_list_train']
                magerr_list_cross = data_dict['magerr_list_cross']
                magerr_list_test = data_dict['magerr_list_test']
                t_list_train = data_dict['t_list_train']
                t_list_cross = data_dict['t_list_cross']
                t_list_test = data_dict['t_list_test']

            # Append Individual Data
            mag_lists_train.append(mag_list_train[0])
            mag_lists_cross.append(mag_list_cross[0])
            mag_lists_test.append(mag_list_test[0])
            magerr_lists_train.append(magerr_list_train[0])
            magerr_lists_cross.append(magerr_list_cross[0])
            magerr_lists_test.append(magerr_list_test[0])
            t_lists_train.append(t_list_train[0])
            t_lists_cross.append(t_list_cross[0])
            t_lists_test.append(t_list_test[0])

        # Save All Data
        print(mag_lists_train)
        mag_lists_train = np.array(mag_lists_train)
        mag_lists_cross = np.array(mag_lists_cross)
        mag_lists_test = np.array(mag_lists_test)
        magerr_lists_train = np.array(magerr_lists_train)
        magerr_lists_cross = np.array(magerr_lists_cross)
        magerr_lists_test = np.array(magerr_lists_test)
        t_lists_train = np.array(t_lists_train)
        t_lists_cross = np.array(t_lists_cross)
        t_lists_test = np.array(t_lists_test)

        all_data_dict = {'mag_lists_train': mag_lists_train, 'mag_lists_cross': mag_lists_cross,
                         'mag_lists_test': mag_lists_test, 'magerr_lists_train': magerr_lists_train,
                         'magerr_lists_cross': magerr_lists_cross, 'magerr_lists_test': magerr_lists_test,
                         't_lists_train': t_lists_train, 't_lists_cross': t_lists_cross,
                         't_lists_test': t_lists_test}

        with open(os.path.join(self.basic_data_path, 'all.pkl'), 'wb') as handle:
            pickle.dump(all_data_dict, handle)

    # ------------------------- LSTM Data Processor -------------------------
    @staticmethod
    def delta_list(raw_list):
        delta_list = []
        for i in range(1, len(raw_list)):
            delta = raw_list[i] - raw_list[i - 1]
            delta_list.append(delta)
        delta_list = np.array(delta_list)

        return delta_list

    def create_X_y(self, scaled_mag_list, scaled_delta_t_list):
        window_len = self.model_config['standard_lstm']['window_len']

        X, y = [], []
        for i in range(1, len(scaled_mag_list) - window_len):
            features = []
            for j in range(i, i + window_len):
                feature = np.concatenate((scaled_mag_list[j], scaled_delta_t_list[j]), axis=0)
                features.append(feature)
            X.append(features)
            y.append(scaled_mag_list[i + window_len])

        X = np.array(X)  # shape=[len(mag_list)-2, window_len, 2]
        y = np.array(y)  # shape=[len(mag_list)-2, 1]

        return X, y

    def prepare_rescale_mag(self):
        for crts_id in self.crts_list:
            with open(os.path.join(self.basic_data_path, str(crts_id) + '.pkl'), 'rb') as handle:
                data_dict = pickle.load(handle)

                # Retrieve Individual Data
                mag_list_train = data_dict['mag_list_train']
                mag_list_cross = data_dict['mag_list_cross']
                mag_list_test = data_dict['mag_list_test']

            # Scale Individual Data
            mag_list = np.concatenate((mag_list_train[0], mag_list_cross[0], mag_list_test[0]), axis=0).reshape(-1, 1)
            print(np.shape(mag_list))
            mag_scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_mag_list = mag_scaler.fit_transform(mag_list)

            train_len = len(mag_list_train)
            cross_len = len(mag_list_cross)

            scaled_mag_list_train = scaled_mag_list[:train_len]
            scaled_mag_list_cross = scaled_mag_list[train_len:(train_len + cross_len)]
            scaled_mag_list_test = scaled_mag_list[(train_len + cross_len):]

            scaled_mag_data_dict = {'scaled_mag_list_train': scaled_mag_list_train,
                                    'scaled_mag_list_cross': scaled_mag_list_cross,
                                    'scaled_mag_list_test': scaled_mag_list_test}

            # Save Scaled Data
            with open(os.path.join(self.standard_lstm_data_path, 'rescaled_mag_' + str(crts_id) + '.pkl'), 'wb') as handle:
                pickle.dump(scaled_mag_data_dict, handle)

            # Save Scaler
            with open(os.path.join(self.standard_lstm_data_path, 'mag_scaler_' + str(crts_id) + '.pkl'), 'wb') as handle:
                joblib.dump(mag_scaler, handle)

    def prepare_rescale_delta_t(self):
        for crts_id in self.crts_list:
            with open(os.path.join(self.basic_data_path, str(crts_id) + '.pkl'), 'rb') as handle:
                data_dict = pickle.load(handle)

                # Retrieve Individual Data
                t_list_train = data_dict['t_list_train']
                t_list_cross = data_dict['t_list_cross']
                t_list_test = data_dict['t_list_test']

                delta_t_list_train = self.delta_list(t_list_train)
                delta_t_list_cross = self.delta_list(t_list_cross)
                delta_t_list_test = self.delta_list(t_list_test)

            # Scale Individual Data
            delta_t_scaler = MinMaxScaler(feature_range=(0, 1))
            delta_t_list = np.concatenate((delta_t_list_train, delta_t_list_cross, delta_t_list_test), axis=0).reshape(
                -1, 1)
            scaled_delta_t_list = delta_t_scaler.fit_transform(delta_t_list)

            train_len = len(t_list_train)
            cross_len = len(t_list_cross)

            scaled_delta_t_list_train = scaled_delta_t_list[:train_len]
            scaled_delta_t_list_cross = scaled_delta_t_list[train_len:(train_len + cross_len)]
            scaled_delta_t_list_test = scaled_delta_t_list[(train_len + cross_len):]

            scaled_delta_t_data_dict = {'scaled_delta_t_list_train': scaled_delta_t_list_train,
                                        'scaled_delta_t_list_cross': scaled_delta_t_list_cross,
                                        'scaled_delta_t_list_test': scaled_delta_t_list_test}

            # Save Scaled Data
            with open(os.path.join(self.standard_lstm_data_path, 'rescaled_delta_t_' + str(crts_id) + '.pkl'), 'wb') as handle:
                pickle.dump(scaled_delta_t_data_dict, handle)

            # Save Scaler
            with open(os.path.join(self.standard_lstm_data_path, 'delta_t_scaler_' + str(crts_id) + '.pkl'), 'wb') as handle:
                joblib.dump(delta_t_scaler, handle)

    def prepare_standard_lstm_data(self):
        for crts_id in self.crts_list:
            # Load Scaled Data
            with open(os.path.join(self.standard_lstm_data_path, 'rescaled_mag' + str(crts_id) + '.pkl'), 'rb') as handle:
                scaled_mag_data_dict = pickle.load(handle)
                scaled_mag_list_train = scaled_mag_data_dict['scaled_mag_list_train']
                scaled_mag_list_cross = scaled_mag_data_dict['scaled_mag_list_cross']
                scaled_mag_list_test = scaled_mag_data_dict['scaled_mag_list_test']

            with open(os.path.join(self.standard_lstm_data_path, 'rescaled_delta_t' + str(crts_id) + '.pkl'), 'rb') as handle:
                scaled_delta_t_data_dict = pickle.load(handle)
                scaled_delta_t_list_train = scaled_delta_t_data_dict['scaled_delta_t_list_train']
                scaled_delta_t_list_cross = scaled_delta_t_data_dict['scaled_delta_t_list_cross']
                scaled_delta_t_list_test = scaled_delta_t_data_dict['scaled_delta_t_list_test']

            train_X, train_y = self.create_X_y(scaled_mag_list_train, scaled_delta_t_list_train)
            cross_X, cross_y = self.create_X_y(scaled_mag_list_cross, scaled_delta_t_list_cross)
            test_X, test_y = self.create_X_y(scaled_mag_list_test, scaled_delta_t_list_test)

            X_y_data_dict = {'train_X': train_X, 'train_y': train_y, 'cross_X': cross_X, 'cross_y': cross_y,
                             'test_X': test_X, 'test_y': test_y}

            # Save X, y Data
            window_len = self.model_config['standard_lstm']['window_len']
            file_name = 'X_y_' + str(crts_id) + '_window_len_' + str(window_len) + '.plk'
            with open(os.path.join(self.standard_lstm_data_path, file_name), 'wb') as handle:
                pickle.dump(X_y_data_dict, handle)

