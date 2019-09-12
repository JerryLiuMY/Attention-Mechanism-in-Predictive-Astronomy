import numpy as np
import os
import json
import pickle
import matplotlib.pyplot as plt
from global_setting import DATA_FOLDER
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from utils.io import delta_list
np.random.seed(1)


class StandardLSTM:
    # Note: This need to be run in python2
    def __init__(self, crts_id):
        # Configuration
        self.crts_id = crts_id
        self.mag_scaler = MinMaxScaler(feature_range=(0, 1))
        self.delta_t_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.load_config()

        # Paths
        self.data_name = str(self.crts_id) + '_window_len_' + str(self.window_len)
        self.model_name = str(self.crts_id) + '_window_len_' + str(self.window_len) + '_hidden_dim_' + self.hidden_dim
        self.data_save_path = os.path.join(DATA_FOLDER, 'processed_data', 'standard_lstm', self.data_name + '.pickle')
        self.model_save_path = os.path.join(DATA_FOLDER, 'model', 'standard_lstm', self.model_name + '.h5')

    def load_config(self):
        self.model_config = json.load(open('./config/model_config.json'))
        self.epochs = self.model_config['standard_lstm']['epochs']
        self.batch_size = self.model_config['standard_lstm']['batch_size']
        self.hidden_dim = self.model_config['standard_lstm']['hidden_dim']
        self.window_len = self.model_config['standard_lstm']['window_len']


    def rescale_mag(self, mag_list_train, mag_list_cross, mag_list_test):
        mag_list = np.concatenate((mag_list_train, mag_list_cross, mag_list_test), axis=0).reshape(-1, 1)
        scaled_mag_list = self.mag_scaler.fit_transform(mag_list)

        train_len = len(mag_list_train)
        cross_len = len(mag_list_cross)

        scaled_mag_list_train = scaled_mag_list[:train_len]
        scaled_mag_list_cross = scaled_mag_list[train_len:(train_len+cross_len)]
        scaled_mag_list_test = scaled_mag_list[(train_len+cross_len):]

        return scaled_mag_list_train, scaled_mag_list_cross, scaled_mag_list_test

    def rescale_delta_t(self, t_list_train, t_list_cross, t_list_test):
        delta_t_list_train = delta_list(t_list_train)
        delta_t_list_cross = delta_list(t_list_cross)
        delta_t_list_test = delta_list(t_list_test)

        self.delta_t_scaler = MinMaxScaler(feature_range=(0, 1))
        delta_t_list = np.concatenate((delta_t_list_train, delta_t_list_cross, delta_t_list_test), axis=0).reshape(-1, 1)
        scaled_delta_t_list = self.delta_t_scaler.fit_transform(delta_t_list)
        # print(delta_t_list)

        train_len = len(t_list_train)
        cross_len = len(t_list_cross)

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
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_dim, input_shape=(self.window_len, 2)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')


    def fit_model(self, train_X, train_y, cross_X, cross_y, test_X, test_y):
        # Train model
        self.model.fit(train_X, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        corss_score = self.model.evaluate(cross_X, cross_y, batch_size=128)
        test_score = self.model.evaluate(test_X, test_y, batch_size=128)

        # Save model
        self.model.save(self.model_save_path)

        return corss_score, test_score

    def one_step_prediction(self, train_X, cross_X, test_X):
        # Input
        lstm_model = load_model(self.model_save_path)

        # Train Interpolation
        scaled_y_inter = lstm_model.predict(train_X)
        y_inter = self.mag_scaler.inverse_transform(scaled_y_inter)

        # Cross Prediction
        scaled_cross_y_pred = lstm_model.predict(cross_X)
        cross_y_pred = self.mag_scaler.inverse_transform(scaled_cross_y_pred)

        # Test Prediction
        scaled_test_y_pred = lstm_model.predict(test_X)
        test_y_pred = self.mag_scaler.inverse_transform(scaled_test_y_pred)

        return y_inter, cross_y_pred, test_y_pred

    def multiple_step_prediction(self, train_X, cross_X, test_X, scaled_cross_delta_t_list):
        # Load Model
        lstm_model = load_model(self.model_save_path)

        # Train Interpolation
        scaled_y_inter = lstm_model.predict(train_X)
        y_inter = self.mag_scaler.inverse_transform(scaled_y_inter)

        # Cross Prediction
        cross_X_step = cross_X[0]
        scaled_y_pred = []
        for i in range(self.window_len, len(scaled_cross_delta_t_list)-1):
            cross_X_step = np.expand_dims(cross_X_step, axis=0)
            scaled_y_pred_step = lstm_model.predict(cross_X_step)
            scaled_y_pred_step = np.squeeze(scaled_y_pred_step, axis=0)
            scaled_y_pred.append(scaled_y_pred_step)
            new_feature = np.concatenate((scaled_y_pred_step, scaled_cross_delta_t_list[i+1]), axis=0)
            new_feature = np.expand_dims(new_feature, axis=0)
            cross_X_step = np.concatenate((cross_X_step[0, 1:, :], new_feature), axis=0)

        scaled_y_pred = np.array(scaled_y_pred)
        cross_y_pred = self.mag_scaler.inverse_transform(scaled_y_pred)

        return y_inter, cross_y_pred, test_y_pred

    def plot_prediction(self, mag_list_train, mag_list_cross, mag_list_test, t_list_cross, t_list_train, t_list_test,
                        magerr_list_train, magerr_list_cross, magerr_list_test, y_inter, cross_y_pred, test_y_pred):
        # Specify parameters
        max_time = t_list_test.max()
        min_time = t_list_train.min()
        upper_limit = max_time
        lower_limit = min_time

        # Plot the function, the prediction and the 95% confidence interval based on the MSE
        plt.figure(figsize=(12, 8))
        plt.errorbar(t_list_train, mag_list_train, magerr_list_train,
                     fmt='k.', markersize=10, label='Training')
        plt.errorbar(t_list_cross, mag_list_cross, magerr_list_cross,
                     fmt='k.', markersize=10, label='Validation')
        plt.errorbar(t_list_test, mag_list_test, magerr_list_test,
                     fmt='k.', markersize=10, label='Test')
        plt.scatter(t_list_train[self.window_len: -1], y_inter, color='g')
        plt.scatter(t_list_cross[self.window_len: -1], cross_y_pred, color='b')
        plt.scatter(t_list_cross[self.window_len: -1], test_y_pred, color='r')
        plt.xlim(lower_limit, upper_limit)
        plt.xlabel('MJD')
        plt.ylabel('Mag')
        plt.legend(loc='upper left')
        plt.title('crts_id: ' + str(self.crts_id))
        plt.show()
