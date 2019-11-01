import numpy as np
import matplotlib.pyplot as plt
import json
from keras.layers import LSTM, Dense
from keras.models import Sequential
from utils.phased_lstm import PhasedLSTM
np.random.seed(1)

class VanillaLSTM:
    # The model is now trained individually for each sample, so we feed in the crts_id for now
    def __init__(self, window_len, hidden_dim, epochs, batch_size, phased=False):
        # Configuration
        self.window_len = window_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.phased = phased
        self.model = None

    def build_model(self):
        # Build Model
        self.model = Sequential()
        if self.phased:
            self.model.add(PhasedLSTM(self.hidden_dim, input_shape=(self.window_len, 2)))
        else:
            self.model.add(LSTM(self.hidden_dim, input_shape=(self.window_len, 2)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit_model(self, train_X, train_y, cross_X, cross_y, test_X, test_y):
        # Train Model
        train_score = self.model.fit(train_X, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        corss_score = self.model.evaluate(cross_X, cross_y, batch_size=self.batch_size)
        test_score = self.model.evaluate(test_X, test_y, batch_size=self.batch_size)

        return self.model

    def multi_step_prediction(self, X_train, X_cross, X_test, mag_scaler):
        # Train Interpolation
        scaled_y_inter = self.model.predict(X_train)
        y_inter = mag_scaler.inverse_transform(scaled_y_inter)

        # Cross Prediction
        scaled_cross_y_pred = self.model.predict(X_cross)
        cross_y_pred = mag_scaler.inverse_transform(scaled_cross_y_pred)

        # Test Prediction
        scaled_test_y_pred = self.model.predict(X_test)
        test_y_pred = mag_scaler.inverse_transform(scaled_test_y_pred)

        return y_inter, cross_y_pred, test_y_pred

    def one_step_prediction(self, X_train, X_cross, X_test, mag_scaler):
        # Train Interpolation
        scaled_y_inter = self.model.predict(X_train)
        y_inter = mag_scaler.inverse_transform(scaled_y_inter)

        # Cross Prediction
        y_pred_cross = self.one_step(X_cross, mag_scaler)

        # Test Prediction
        y_pred_test = self.one_step(X_test, mag_scaler)

        # Plot Prediction

        return fig

    def one_step(self, X, mag_scaler):
        # Cross Prediction
        scaled_y = []
        X_step = X[[0]]  # shape = (1, window_len, num_feature)
        print(np.shape(X))
        for i in range(np.shape(X)[0]):
            scaled_y_step = self.model.predict(X_step)  # shape = (1, num_feature)
            scaled_y.append(scaled_y_step[0])  # scaled_y_step[0] -> shape = (num_feature)
            if i == np.shape(X)[0]-1:
                break

            scaled_t_step = X[i+1][[-1], -1:]  # shape = (1, 1)
            new_feature = np.concatenate((scaled_y_step, scaled_t_step), axis=1)  # shape = (1, num_feature + 1)
            X_step = np.concatenate((X_step[0][1:, :], new_feature), axis=0)  # shape = (window_len, num_feature + 1)
            X_step = np.expand_dims(X_step, axis=0)  # shape = (1, window_len, num_feature + 1)

        scaled_y = np.array(scaled_y)  # shape = (num_step, num_feature)
        y = mag_scaler.inverse_transform(scaled_y)  # shape = (num_step, num_feature)

        return y

    def plot_prediction(self, mag_list_train, mag_list_cross, mag_list_test, t_list_cross, t_list_train, t_list_test,
                        magerr_list_train, magerr_list_cross, magerr_list_test, y_inter, y_pred_cross, y_pred_test):
        # Specify parameters
        max_time = t_list_test.max()
        min_time = t_list_train.min()
        upper_limit = max_time
        lower_limit = min_time

        # Plot the function, the prediction and the 95% confidence interval based on the MSE
        fig = plt.figure(figsize=(12, 8))
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
        plt.title('Simulated Path')
        plt.show()

        return fig
