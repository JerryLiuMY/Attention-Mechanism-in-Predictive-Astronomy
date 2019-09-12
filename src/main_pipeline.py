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
        self.crts_id = crts_id
        self.data_config = json.load(open('./config/data_config.json'))
        self.model_config = json.load(open('./config/model_config.json'))
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

    def standard_lstm(self):
        standard_lstm = StandardLSTM()
        scaled_mag_list_train, scaled_mag_list_cross, scaled_mag_list_test = standard_lstm.rescale_mag()


    def rescale_mag(self):
        self.mag_scaler = MinMaxScaler(feature_range=(0, 1))
        mag_list = np.concatenate((self.mag_list_train, self.mag_list_cross, self.mag_list_test), axis=0).reshape(-1, 1)
        scaled_mag_list = self.mag_scaler.fit_transform(mag_list)

        train_len = len(self.mag_list_train)
        cross_len = len(self.mag_list_cross)

        scaled_mag_list_train = scaled_mag_list[:train_len]
        scaled_mag_list_cross = scaled_mag_list[train_len:(train_len+cross_len)]
        scaled_mag_list_test = scaled_mag_list[(train_len+cross_len):]

        return scaled_mag_list_train, scaled_mag_list_cross, scaled_mag_list_test


    def rescale_delta_t(self):
        delta_t_list_train = delta_list(self.t_list_train)
        delta_t_list_cross = delta_list(self.t_list_cross)
        delta_t_list_test = delta_list(self.t_list_test)

        self.delta_t_scaler = MinMaxScaler(feature_range=(0, 1))
        delta_t_list = np.concatenate((delta_t_list_train, delta_t_list_cross, delta_t_list_test), axis=0).reshape(-1, 1)
        scaled_delta_t_list = self.delta_t_scaler.fit_transform(delta_t_list)
        # print(delta_t_list)

        train_len = len(self.t_list_train)
        cross_len = len(self.t_list_cross)

        scaled_delta_t_list_train = scaled_delta_t_list[:train_len]
        scaled_delta_t_list_cross = scaled_delta_t_list[train_len:(train_len+cross_len)]
        scaled_delta_t_list_test = scaled_delta_t_list[(train_len+cross_len):]

        return scaled_delta_t_list_train, scaled_delta_t_list_cross, scaled_delta_t_list_test

    def prepare_data(self, scaled_mag_list, scaled_delta_t_list):
        X, y = [], []
        for i in range(1, len(scaled_mag_list) - self.window_len):
            features = []
            for j in range(i, i + self.window_len):
                feature = np.concatenate((scaled_mag_list[j], scaled_delta_t_list[j]), axis=0)
                features.append(feature)
            X.append(features)
            y.append(scaled_mag_list[i + self.window_len])

        X = np.array(X)  # shape=[len(mag_list)-2, window_len, 2]
        y = np.array(y)  # shape=[len(mag_list)-2, 1]

        return X, y

    def build_model(self):
        # Build model
        lstm_model = Sequential()
        lstm_model.add(LSTM(4, input_shape=(self.model_config['standard_lstm']['window_len'], 2)))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')

        return lstm_model

    def fit_model(self):
        # Input
        scaled_mag_list_train, scaled_mag_list_cross, scaled_mag_list_test = self.rescale_mag()
        scaled_delta_t_list_train, scaled_delta_t_list_cross, scaled_delta_t_list_test = self.rescale_delta_t()

        # Load configuration
        epochs = self.model_config['standard_lstm']['epochs']
        batch_size = self.model_config['standard_lstm']['batch_size']
        lstm_model = self.build_model()

        # Load data
        train_X, train_y = self.prepare_data(scaled_mag_list_train, scaled_delta_t_list_train)
        cross_X, cross_y = self.prepare_data(scaled_mag_list_cross, scaled_delta_t_list_cross)

        # Train model
        lstm_model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=2)
        score = lstm_model.evaluate(cross_X, cross_y, batch_size=128)

        # Save model
        lstm_model.save(self.model_save_path)

        return score

    def one_step_prediction(self):
        # Input
        scaled_mag_list_train, scaled_mag_list_cross, scaled_mag_list_test = self.rescale_mag()
        scaled_delta_t_list_train, scaled_delta_t_list_cross, scaled_delta_t_list_test = self.rescale_delta_t()
        lstm_model = load_model(self.model_save_path)

        train_X, train_y = self.prepare_data(scaled_mag_list_train, scaled_delta_t_list_train)
        scaled_y_inter = lstm_model.predict(train_X)
        y_inter = self.mag_scaler.inverse_transform(scaled_y_inter)

        cross_X, cross_y = self.prepare_data(scaled_mag_list_cross, scaled_delta_t_list_cross)
        scaled_y_pred = lstm_model.predict(cross_X)
        y_pred = self.mag_scaler.inverse_transform(scaled_y_pred)
        self.plot_prediction(y_inter, y_pred)

    def multiple_step_prediction(self):
        # Input
        scaled_mag_list_train, scaled_mag_list_cross, scaled_mag_list_test = self.rescale_mag()
        scaled_delta_t_list_train, scaled_delta_t_list_cross, scaled_delta_t_list_test = self.rescale_delta_t()
        lstm_model = load_model(self.model_save_path)

        # Training Interpolation
        train_X, train_y = self.prepare_data(scaled_mag_list_train, scaled_delta_t_list_train)
        scaled_y_inter = lstm_model.predict(train_X)
        y_inter = self.mag_scaler.inverse_transform(scaled_y_inter)

        # Cross Prediction
        cross_X, cross_y = self.prepare_data(scaled_mag_list_cross, scaled_delta_t_list_cross)
        cross_X_step = cross_X[0]
        scaled_y_pred = []
        for i in range(self.window_len, len(scaled_delta_t_list_cross)-1):
            cross_X_step = np.expand_dims(cross_X_step, axis=0)
            scaled_y_pred_step = lstm_model.predict(cross_X_step)
            scaled_y_pred_step = np.squeeze(scaled_y_pred_step, axis=0)
            scaled_y_pred.append(scaled_y_pred_step)
            new_feature = np.concatenate((scaled_y_pred_step, scaled_delta_t_list_cross[i+1]), axis=0)
            new_feature = np.expand_dims(new_feature, axis=0)
            cross_X_step = np.concatenate((cross_X_step[0, 1:, :], new_feature), axis=0)

        scaled_y_pred = np.array(scaled_y_pred)
        y_pred = self.mag_scaler.inverse_transform(scaled_y_pred)
        self.plot_prediction(y_inter, y_pred)

    def plot_prediction(self, y_inter, y_pred):
        # Specify parameters
        max_time = self.t_list_test.max()
        min_time = self.t_list_train.min()
        upper_limit = max_time
        lower_limit = min_time

        # Plot the function, the prediction and the 95% confidence interval based on the MSE
        plt.figure(figsize=(12, 8))
        plt.errorbar(self.t_list_train, self.mag_list_train, self.magerr_list_train,
                     fmt='k.', markersize=10, label='Training')
        plt.errorbar(self.t_list_cross, self.mag_list_cross, self.magerr_list_cross,
                     fmt='k.', markersize=10, label='Validation')
        plt.errorbar(self.t_list_test, self.mag_list_test, self.magerr_list_test,
                     fmt='k.', markersize=10, label='Test')
        plt.scatter(self.t_list_train[self.window_len: -1], y_inter, color='g')
        plt.scatter(self.t_list_cross[self.window_len: -1], y_pred, color='b')
        plt.xlim(lower_limit, self.t_list_cross.max())
        plt.xlabel('MJD')
        plt.ylabel('Mag')
        plt.legend(loc='upper left')
        plt.title('crts_id: ' + str(self.crts_id))
        plt.show()
