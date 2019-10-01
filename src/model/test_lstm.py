import numpy as np
import os
import matplotlib.pyplot as plt
import json
from keras.layers import Dense
from keras.models import Sequential, load_model
from global_setting import standard_lstm_model_folder, phased_lstm_model_folder
from utils.test_lstm_layer import LSTM
np.random.seed(1)

class TestLSTM:
    # The model is now trained individually for each sample, so we feed in the crts_id for now
    def __init__(self, crts_id, phased=False):
        # Configuration
        self.crts_id = crts_id
        self.phased = phased
        self.model = None
        self.load_config()

        # Model Path & Name
        self.model_name = '{}_window_len_{}_hidden_dim_{}'.format(str(self.crts_id), str(self.window_len), str(self.hidden_dim))
        self.standard_lstm_model_path = os.path.join(standard_lstm_model_folder, self.model_name + '.h5')
        self.phased_lstm_model_path = os.path.join(phased_lstm_model_folder, self.model_name + '.h5')

    def load_config(self):
        self.model_config = json.load(open('./config/model_config.json'))
        self.epochs = self.model_config['basic_lstm']['epochs']
        self.batch_size = self.model_config['basic_lstm']['batch_size']
        self.hidden_dim = self.model_config['basic_lstm']['hidden_dim']
        self.window_len = self.model_config['basic_lstm']['window_len']

    def build_model(self):
        # Build Model
        self.model = Sequential()
        if self.phased:
            self.model.add(LSTM(self.hidden_dim, input_shape=(self.window_len, 2)))
        else:
            self.model.add(LSTM(self.hidden_dim, input_shape=(self.window_len, 2)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit_model(self, train_X, train_y, cross_X, cross_y, test_X, test_y):
        # Train Model
        train_score = self.model.fit(train_X, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        corss_score = self.model.evaluate(cross_X, cross_y, batch_size=self.batch_size)
        test_score = self.model.evaluate(test_X, test_y, batch_size=self.batch_size)

        # Save model
        if self.phased:
            self.model.save(self.phased_lstm_model_path)
        else:
            self.model.save(self.standard_lstm_model_path)

        return train_score, corss_score, test_score

    def one_step_prediction(self, train_X, cross_X, test_X, mag_scaler):
        # Load Model
        if self.phased:
            lstm_model = load_model(self.phased_lstm_model_path)
        else:
            lstm_model = load_model(self.standard_lstm_model_path)

        # Train Interpolation
        scaled_y_inter = lstm_model.predict(train_X)
        y_inter = mag_scaler.inverse_transform(scaled_y_inter)

        # Cross Prediction
        scaled_cross_y_pred = lstm_model.predict(cross_X)
        cross_y_pred = mag_scaler.inverse_transform(scaled_cross_y_pred)

        # Test Prediction
        scaled_test_y_pred = lstm_model.predict(test_X)
        test_y_pred = mag_scaler.inverse_transform(scaled_test_y_pred)

        return y_inter, cross_y_pred, test_y_pred

    def multiple_step_prediction(self, train_X, cross_X, test_X, mag_scaler):
        # Load Model
        if self.phased:
            lstm_model = load_model(self.phased_lstm_model_path)
        else:
            lstm_model = load_model(self.standard_lstm_model_path)

        # Train Interpolation
        scaled_y_inter = lstm_model.predict(train_X)
        y_inter = mag_scaler.inverse_transform(scaled_y_inter)

        # Cross Prediction
        scaled_y_pred_cross = []
        X_cross_step = cross_X[[0]]
        print(np.shape(cross_X))
        for i in range(np.shape(cross_X)[0]):
            scaled_y_pred_cross_step = lstm_model.predict(X_cross_step)  # shape = (1, num_feature)
            scaled_y_pred_cross.append(scaled_y_pred_cross_step[0])
            if i == np.shape(cross_X)[0]-1:
                break

            scaled_cross_t_step = cross_X[i+1][[-1], -1:]  # shape = (1, 1)
            new_feature = np.concatenate((scaled_y_pred_cross_step, scaled_cross_t_step), axis=1)  # shape = (1, num_feature + 1)
            X_cross_step = np.concatenate((X_cross_step[0][1:, :], new_feature), axis=0)  # shape = (window_len, num_feature + 1)
            X_cross_step = np.expand_dims(X_cross_step, axis=0)  # shape = (1, window_len, num_feature + 1)

        scaled_y_pred_cross = np.array(scaled_y_pred_cross)
        y_pred_cross = mag_scaler.inverse_transform(scaled_y_pred_cross)

        # Test Prediction
        scaled_y_pred_test = []
        X_test_step = test_X[[0]]
        for i in range(np.shape(test_X)[0]):
            scaled_y_pred_test_step = lstm_model.predict(X_test_step)  # shape = (1, num_feature)
            scaled_y_pred_test.append(scaled_y_pred_test_step[0])
            if i == np.shape(test_X)[0]-1:
                break

            scaled_test_t_step = test_X[i+1][[-1], -1:]  # shape = (1, 1)
            new_feature = np.concatenate((scaled_y_pred_test_step, scaled_test_t_step), axis=1)  # shape = (1, num_feature + 1)
            X_test_step = np.concatenate((X_test_step[0][1:, :], new_feature), axis=0)  # shape = (window_len, num_feature + 1)
            X_test_step = np.expand_dims(X_test_step, axis=0)  # shape = (1, window_len, num_feature + 1)


        scaled_y_pred_test = np.array(scaled_y_pred_test)
        y_pred_test = mag_scaler.inverse_transform(scaled_y_pred_test)

        return y_inter, y_pred_cross, y_pred_test

    def plot_prediction(self, mag_list_train, mag_list_cross, mag_list_test, t_list_cross, t_list_train, t_list_test,
                        magerr_list_train, magerr_list_cross, magerr_list_test, y_inter, y_pred_cross, y_pred_test):
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
        plt.scatter(t_list_train[self.window_len+1: -1], y_inter, color='g')
        plt.scatter(t_list_cross[self.window_len+1: -1], y_pred_cross, color='b')
        plt.scatter(t_list_test[self.window_len+1: -1], y_pred_test, color='r')
        plt.xlim(lower_limit, upper_limit)
        plt.xlabel('MJD')
        plt.ylabel('Mag')
        plt.legend(loc='upper left')
        plt.title('crts_id: ' + str(self.crts_id))
        plt.show()
