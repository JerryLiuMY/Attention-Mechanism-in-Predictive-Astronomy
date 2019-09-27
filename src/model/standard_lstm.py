import numpy as np
import os
import json
import matplotlib.pyplot as plt
from global_setting import DATA_FOLDER
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
np.random.seed(1)


class StandardLSTM:
    def __init__(self, crts_id):
        # Configuration
        self.crts_id = crts_id
        self.model = None
        self.load_config()

        # Paths
        self.model_name = 'model_{}_window_len_{}__hidden_dim_{}'.format(str(self.crts_id), str(self.window_len), str(self.hidden_dim))
        self.model_path = os.path.join(DATA_FOLDER, 'model', 'standard_lstm', self.model_name + '.h5')

    def load_config(self):
        self.model_config = json.load(open('./config/model_config.json'))
        self.epochs = self.model_config['standard_lstm']['epochs']
        self.batch_size = self.model_config['standard_lstm']['batch_size']
        self.hidden_dim = self.model_config['standard_lstm']['hidden_dim']
        self.window_len = self.model_config['standard_lstm']['window_len']

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_dim, input_shape=(self.window_len, 2)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit_model(self, train_X, train_y, cross_X, cross_y, test_X, test_y):
        train_score = self.model.fit(train_X, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        corss_score = self.model.evaluate(cross_X, cross_y, batch_size=128)
        test_score = self.model.evaluate(test_X, test_y, batch_size=128)

        # Save model
        self.model.save(self.model_path)

        return train_score, corss_score, test_score

    def one_step_prediction(self, train_X, cross_X, test_X, mag_scaler):
        # Input
        lstm_model = load_model(self.model_path)

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

    def multiple_step_prediction(self, train_X, cross_X, test_X, scaled_delta_t_list_cross, scaled_delta_t_list_test, mag_scaler):
        # Load Model
        lstm_model = load_model(self.model_path)

        # Train Interpolation
        scaled_y_inter = lstm_model.predict(train_X)
        y_inter = mag_scaler.inverse_transform(scaled_y_inter)

        # Cross Prediction
        cross_X_step = cross_X[0]
        scaled_y_pred_cross = []
        for i in range(self.window_len, len(scaled_delta_t_list_cross)-1):
            cross_X_step = np.expand_dims(cross_X_step, axis=0)
            scaled_cross_y_pred_step = lstm_model.predict(cross_X_step)
            scaled_y_pred_cross_step = np.squeeze(scaled_cross_y_pred_step, axis=0)
            scaled_y_pred_cross.append(scaled_y_pred_cross_step)
            new_feature = np.concatenate((scaled_cross_y_pred_step, scaled_delta_t_list_cross[i+1]), axis=0)
            new_feature = np.expand_dims(new_feature, axis=0)
            cross_X_step = np.concatenate((cross_X_step[0, 1:, :], new_feature), axis=0)  # TODO: Dataset

        scaled_y_pred_cross = np.array(scaled_y_pred_cross)
        y_pred_cross = mag_scaler.inverse_transform(scaled_y_pred_cross)

        # Test Prediction
        test_X_step = test_X[0]
        scaled_y_pred_test = []
        for i in range(self.window_len, len(scaled_delta_t_list_test)-1):
            test_X_step = np.expand_dims(test_X_step, axis=0)
            scaled_y_pred_test_step = lstm_model.predict(test_X_step)
            scaled_y_pred_test_step = np.squeeze(scaled_y_pred_test_step, axis=0)
            scaled_y_pred_test.append(scaled_y_pred_test_step)
            new_feature = np.concatenate((scaled_y_pred_test_step, scaled_delta_t_list_test[i+1]), axis=0)
            new_feature = np.expand_dims(new_feature, axis=0)
            test_X_step = np.concatenate((test_X_step[0, 1:, :], new_feature), axis=0)  # TODO: Dataset

        scaled_y_pred_test = np.array(scaled_y_pred_test)
        y_pred_test = mag_scaler.inverse_transform(scaled_y_pred_test)

        return y_inter, y_pred_cross, y_pred_test

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
