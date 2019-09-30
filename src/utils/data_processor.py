import math
import numpy as np
import pandas as pd
import os
import json
import joblib
import pickle
from global_setting import DATA_FOLDER
from global_setting import raw_data_path, basic_data_path, standard_lstm_data_path, phased_lstm_data_path
from sklearn.preprocessing import MinMaxScaler
np.random.seed(1)


class BasicDataProcessor:
    def __init__(self, crts_id):
        # crts_id
        self.crts_id = crts_id

        # Configuration
        self.load_config()

        # Data Path
        self.raw_data_path = raw_data_path
        self.basic_data_path = basic_data_path
        self.standard_lstm_data_path = standard_lstm_data_path

        # Data Name
        self.basic_data_name = str(self.crts_id) + '.pkl'

        # self.prepare_basic_data()
        # inputs: len(content['Mag']) = (num_data, )
        # inputs: len(content['Magerr']) = (num_data, )
        # inputs: len(content['MJD']) = (num_data, )
        # output: shape(mag_list_train) = (num_train_data, )
        # output: shape(mag_cross_train) = (num_cross_data, )
        # output: shape(mag_test_train) = (num_test_data, )

    def load_config(self):
        self.data_config = json.load(open('./config/data_config.json'))
        self.train_percent = self.data_config['data_loader']['train_partition']
        self.cross_percent = self.data_config['data_loader']['cross_partition']
        self.model_config = json.load(open('./config/model_config.json'))

    def partition_index(self, length):
        p1 = int(math.floor(length * self.train_percent))
        p2 = int(math.floor(length * (self.train_percent + self.cross_percent)))

        index_list = list(range(length))
        index_train = np.array(index_list[0:p1])
        index_cross = np.array(index_list[p1:p2])
        index_test = np.array(index_list[p2:])

        return index_train, index_cross, index_test

    def prepare_basic_data(self):
        with open(os.path.join(self.raw_data_path, str(self.crts_id) + '.csv')) as handle:
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

        data_dict = {'mag_list_train': mag_list_train,
                     'mag_list_cross': mag_list_cross,
                     'mag_list_test': mag_list_test,
                     'magerr_list_train': magerr_list_train,
                     'magerr_list_cross': magerr_list_cross,
                     'magerr_list_test': magerr_list_test,
                     't_list_train': t_list_train,
                     't_list_cross': t_list_cross,
                     't_list_test': t_list_test}

        with open(os.path.join(self.basic_data_path, self.basic_data_name), 'wb') as handle:
            pickle.dump(data_dict, handle)

class CarmaDataProcessor:
    def __init__(self):
        # Dimensions
        # inputs: basic data
        # output: t_shape = (num_steps, )
        # output: y_shape = (num_steps, )
        # output: yerr_shape = (num_steps, )
        pass
    pass

class GPDataProcessor:
    def __init__(self):

        # prepare_rescale_mag():
        # inputs: basic data
        # output: X_shape = (num_steps, num_features)
        # output: y_shape = (num_steps, )
        pass
    pass

class LSTMDataProcessor:

    def __init__(self, crts_id):
        # crts_id
        self.crts_id = crts_id

        # Configuration
        self.load_config()

        # Data Paths
        self.raw_data_path = raw_data_path
        self.basic_data_path = basic_data_path
        self.standard_lstm_data_path = standard_lstm_data_path
        self.phased_lstm_data_path = phased_lstm_data_path

        # Basic Data Name
        self.basic_data_name = str(self.crts_id) + '.pkl'

        # LSTM Data Name
        self.rescaled_mag_name = str(self.crts_id) + '_rescaled_mag' + '.pkl'
        self.mag_scaler_name = str(self.crts_id) + '_mag_scaler' + '.pkl'
        self.rescaled_delta_t_name = str(self.crts_id) + '_rescaled_delta_t' + '.pkl'
        self.delta_t_scaler_name = str(self.crts_id) + '_delta_t_scaler' + '.pkl'

        standard_lstm_window_len = self.model_config['phased_lstm']['window_len']
        self.standard_X_y_name = str(self.crts_id) + '_X_y' + '_window_len_' + str(standard_lstm_window_len) + '.plk'

        phased_lstm_window_len = self.model_config['phased_lstm']['window_len']
        self.phased_X_y_name = str(self.crts_id) + '_X_y' + '_window_len_' + str(phased_lstm_window_len) + '.plk'

        # self.prepare_rescale_mag()
        # input: shape(mag_list_train) = (num_train_data, )
        # input: shape(mag_list_cross) = (num_cross_data, )
        # input: shape(mag_list_test) = (num_test_data, )
        # output: shape(scaled_mag_list_train) = (num_train_data - 1, 1)
        # output: shape(scaled_mag_list_cross) = (num_cross_data - 1, 1)
        # output: shape(scaled_mag_list_test) = (num_test_data - 1, 1)

        # self.prepare_rescale_delta_t()
        # input: shape(t_list_train) = (num_train_data, )
        # input: shape(t_list_cross) = (num_cross_data, )
        # input: shape(t_list_test) = (num_test_data, )
        # output: shape(scaled_delta_t_list_train) = (num_train_data - 1, 1)
        # output: shape(scaled_delta_t_list_cross) = (num_cross_data - 1, 1)
        # output: shape(scaled_delta_t_list_test) = (num_test_data - 1, 1)

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

    def load_config(self):
        self.data_config = json.load(open('./config/data_config.json'))
        self.train_percent = self.data_config['data_loader']['train_partition']
        self.cross_percent = self.data_config['data_loader']['cross_partition']
        self.model_config = json.load(open('./config/model_config.json'))

    @staticmethod
    def delta_list(raw_list):
        delta_list = []
        for i in range(1, len(raw_list)):
            delta = raw_list[i] - raw_list[i - 1]
            delta_list.append(delta)
        delta_list = np.array(delta_list)

        return delta_list

    def prepare_rescale_mag(self):
        with open(os.path.join(self.basic_data_path, self.basic_data_name), 'rb') as handle:
            data_dict = pickle.load(handle)

            # Retrieve Individual Data
            mag_list_train = data_dict['mag_list_train'][1:].reshape(-1, 1)
            mag_list_cross = data_dict['mag_list_cross'][1:].reshape(-1, 1)
            mag_list_test = data_dict['mag_list_test'][1:].reshape(-1, 1)

        # Scale Individual Data
        mag_list = np.concatenate((mag_list_train, mag_list_cross, mag_list_test), axis=0)
        mag_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_mag_list = mag_scaler.fit_transform(mag_list)

        train_len = np.shape(mag_list_train)[0]
        cross_len = np.shape(mag_list_cross)[0]

        scaled_mag_list_train = scaled_mag_list[:train_len]
        scaled_mag_list_cross = scaled_mag_list[train_len:(train_len + cross_len)]
        scaled_mag_list_test = scaled_mag_list[(train_len + cross_len):]

        scaled_mag_data_dict = {'scaled_mag_list_train': scaled_mag_list_train,
                                'scaled_mag_list_cross': scaled_mag_list_cross,
                                'scaled_mag_list_test': scaled_mag_list_test}

        # Save Scaled Data
        with open(os.path.join(self.standard_lstm_data_path, self.rescaled_mag_name), 'wb') as handle:
            pickle.dump(scaled_mag_data_dict, handle)
        with open(os.path.join(self.phased_lstm_data_path, self.rescaled_mag_name), 'wb') as handle:
            pickle.dump(scaled_mag_data_dict, handle)

        # Save Scaler

        with open(os.path.join(self.standard_lstm_data_path, self.mag_scaler_name), 'wb') as handle:
            joblib.dump(mag_scaler, handle)
        with open(os.path.join(self.phased_lstm_data_path, self.mag_scaler_name), 'wb') as handle:
            joblib.dump(mag_scaler, handle)

        # print(np.shape(data_dict['mag_list_train'][1:]),
        #       np.shape(data_dict['mag_list_cross'][1:]),
        #       np.shape(data_dict['mag_list_test'][1:]))
        # print(np.shape(mag_list_train), np.shape(mag_list_cross), np.shape(mag_list_test))
        # print(np.shape(scaled_mag_list_train), np.shape(scaled_mag_list_cross), np.shape(scaled_mag_list_test))

    def prepare_rescale_delta_t(self):
        with open(os.path.join(self.basic_data_path, self.basic_data_name), 'rb') as handle:
            data_dict = pickle.load(handle)

            # Retrieve Individual Data
            delta_t_list_train = self.delta_list(data_dict['t_list_train']).reshape(-1, 1)
            delta_t_list_cross = self.delta_list(data_dict['t_list_cross']).reshape(-1, 1)
            delta_t_list_test = self.delta_list(data_dict['t_list_test']).reshape(-1, 1)

        # Scale Individual Data
        delta_t_list = np.concatenate((delta_t_list_train, delta_t_list_cross, delta_t_list_test), axis=0)
        delta_t_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_delta_t_list = delta_t_scaler.fit_transform(delta_t_list)

        train_len = np.shape(delta_t_list_train)[0]
        cross_len = np.shape(delta_t_list_cross)[0]

        scaled_delta_t_list_train = scaled_delta_t_list[:train_len]
        scaled_delta_t_list_cross = scaled_delta_t_list[train_len:(train_len + cross_len)]
        scaled_delta_t_list_test = scaled_delta_t_list[(train_len + cross_len):]

        scaled_delta_t_data_dict = {'scaled_delta_t_list_train': scaled_delta_t_list_train,
                                    'scaled_delta_t_list_cross': scaled_delta_t_list_cross,
                                    'scaled_delta_t_list_test': scaled_delta_t_list_test}

        # Save Scaled Data
        with open(os.path.join(self.standard_lstm_data_path, self.rescaled_delta_t_name), 'wb') as handle:
            pickle.dump(scaled_delta_t_data_dict, handle)
        with open(os.path.join(self.phased_lstm_data_path, self.rescaled_delta_t_name), 'wb') as handle:
            pickle.dump(scaled_delta_t_data_dict, handle)

        # Save Scaler
        with open(os.path.join(self.standard_lstm_data_path, self.delta_t_scaler_name), 'wb') as handle:
            joblib.dump(delta_t_scaler, handle)
        with open(os.path.join(self.phased_lstm_data_path, self.delta_t_scaler_name), 'wb') as handle:
            joblib.dump(delta_t_scaler, handle)

        # print(np.shape(self.delta_list(data_dict['t_list_train'])),
        #       np.shape(self.delta_list(data_dict['t_list_cross'])),
        #       np.shape(self.delta_list(data_dict['t_list_test'])))
        # print(np.shape(mag_list_train), np.shape(mag_list_cross), np.shape(mag_list_test))
        # print(np.shape(scaled_mag_list_train), np.shape(scaled_mag_list_cross), np.shape(scaled_mag_list_test))

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

        # print(np.shape(scaled_mag_list), np.shape(scaled_delta_t_list))
        # print(np.shape(X), np.shape(y))

        return X, y

    def prepare_standard_lstm_data(self):
        # Load Scaled Data
        with open(os.path.join(self.standard_lstm_data_path, self.rescaled_mag_name), 'rb') as handle:
            scaled_mag_data_dict = pickle.load(handle)
            scaled_mag_list_train = scaled_mag_data_dict['scaled_mag_list_train']
            scaled_mag_list_cross = scaled_mag_data_dict['scaled_mag_list_cross']
            scaled_mag_list_test = scaled_mag_data_dict['scaled_mag_list_test']

        with open(os.path.join(self.standard_lstm_data_path, self.rescaled_delta_t_name), 'rb') as handle:
            scaled_delta_t_data_dict = pickle.load(handle)
            scaled_delta_t_list_train = scaled_delta_t_data_dict['scaled_delta_t_list_train']
            scaled_delta_t_list_cross = scaled_delta_t_data_dict['scaled_delta_t_list_cross']
            scaled_delta_t_list_test = scaled_delta_t_data_dict['scaled_delta_t_list_test']

        window_len = self.model_config['standard_lstm']['window_len']
        X_train, y_train = self.create_X_y(scaled_mag_list_train, scaled_delta_t_list_train, window_len)
        X_cross, y_cross = self.create_X_y(scaled_mag_list_cross, scaled_delta_t_list_cross, window_len)
        X_test, y_test = self.create_X_y(scaled_mag_list_test, scaled_delta_t_list_test, window_len)

        X_y_data_dict = {'train_X': X_train, 'train_y': y_train, 'cross_X': X_cross, 'cross_y': y_cross,
                         'test_X': X_test, 'test_y': y_test}

        # Save X, y Data
        with open(os.path.join(self.standard_lstm_data_path, self.standard_X_y_name), 'wb') as handle:
            pickle.dump(X_y_data_dict, handle)

        # print(np.shape(X_train), np.shape(y_train))
        # print(np.shape(X_cross), np.shape(y_cross))
        # print(np.shape(X_test), np.shape(y_test))

    def prepare_phased_lstm_data(self):
        # Load Scaled Data
        with open(os.path.join(self.phased_lstm_data_path, self.rescaled_mag_name), 'rb') as handle:
            scaled_mag_data_dict = pickle.load(handle)
            scaled_mag_list_train = scaled_mag_data_dict['scaled_mag_list_train']
            scaled_mag_list_cross = scaled_mag_data_dict['scaled_mag_list_cross']
            scaled_mag_list_test = scaled_mag_data_dict['scaled_mag_list_test']

        with open(os.path.join(self.phased_lstm_data_path, self.rescaled_delta_t_name), 'rb') as handle:
            scaled_delta_t_data_dict = pickle.load(handle)
            scaled_delta_t_list_train = scaled_delta_t_data_dict['scaled_delta_t_list_train']
            scaled_delta_t_list_cross = scaled_delta_t_data_dict['scaled_delta_t_list_cross']
            scaled_delta_t_list_test = scaled_delta_t_data_dict['scaled_delta_t_list_test']

        window_len = self.model_config['phased_lstm']['window_len']
        X_train, y_train = self.create_X_y(scaled_mag_list_train, scaled_delta_t_list_train, window_len)
        X_cross, y_cross = self.create_X_y(scaled_mag_list_cross, scaled_delta_t_list_cross, window_len)
        X_test, y_test = self.create_X_y(scaled_mag_list_test, scaled_delta_t_list_test, window_len)

        X_y_data_dict = {'train_X': X_train, 'train_y': y_train, 'cross_X': X_cross, 'cross_y': y_cross,
                         'test_X': X_test, 'test_y': y_test}

        # Save X, y Data
        with open(os.path.join(self.phased_lstm_data_path, self.phased_X_y_name), 'wb') as handle:
            pickle.dump(X_y_data_dict, handle)

        # print(np.shape(X_train), np.shape(y_train))
        # print(np.shape(X_cross), np.shape(y_cross))
        # print(np.shape(X_test), np.shape(y_test))

if __name__ == 'main':
    basic_data_processor = BasicDataProcessor(1001115026824)
    basic_data_processor.prepare_basic_data()

    lstm_data_processor = LSTMDataProcessor(1001115026824)
    lstm_data_processor.prepare_rescale_mag()
    lstm_data_processor.prepare_rescale_delta_t()
    lstm_data_processor.prepare_standard_lstm_data()
    lstm_data_processor.prepare_phased_lstm_data()
