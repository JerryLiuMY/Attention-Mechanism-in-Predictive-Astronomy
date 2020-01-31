import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras import Input, Model
from utils.phased_lstm import PhasedLSTM

np.random.seed(1)


class VanillaLSTM:
    # The model is now trained individually for each sample, so we feed in the crts_id for now
    def __init__(self, window_len, hidden_dim, epochs, batch_size, phased, **kwargs):
        # Configuration
        self.window_len = window_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.phased = phased

    def build_model(self, X_train, y_train):
        inputs = Input(shape=(self.window_len, 2, ))
        if self.phased == 'phased':
            inter = PhasedLSTM(self.hidden_dim)(inputs, training=True)
        else:
            inter = LSTM(self.hidden_dim)(inputs, training=True)

        inter = Dense(self.hidden_dim, activation='tanh')(inter)
        outputs = Dense(1, activation='linear')(inter)

        model = Model(inputs, outputs)
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2)

        return model

    def single_step_prediction(self, t_list_train, mag_list_train, magerr_list_train,
                               t_list_cross, mag_list_cross, magerr_list_cross,
                               X_train, X_cross, X_test, mag_scaler, model):
        # Train Interpolation
        scaled_y_inter_train = model.predict(X_train)
        y_inter_train = mag_scaler.inverse_transform(scaled_y_inter_train)
        train_loss = mean_squared_error(y_inter_train[:, 0], mag_list_train[self.window_len + 1: -1])

        # Cross Prediction
        scaled_y_cross_pred = model.predict(X_cross)
        single_y_pred_cross = mag_scaler.inverse_transform(scaled_y_cross_pred)
        single_cross_loss = mean_squared_error(single_y_pred_cross[:, 0], mag_list_cross[self.window_len+1: -1])

        single_fit_fig = self.plot_prediction(t_list_train, mag_list_train, magerr_list_train,
                                              t_list_cross, mag_list_cross, magerr_list_cross,
                                              y_inter_train, single_y_pred_cross)

        single_res_fig = self.plot_residual(t_list_train, mag_list_train, magerr_list_train,
                                            t_list_cross, mag_list_cross, magerr_list_cross,
                                            y_inter_train, single_y_pred_cross)

        return single_train_loss, single_cross_loss, single_fit_fig, single_res_fig

    def multi_step_prediction(self, t_list_train, mag_list_train, magerr_list_train,
                              t_list_cross, mag_list_cross, magerr_list_cross,
                              X_train, X_cross, X_test, mag_scaler, model):

        # Cross Prediction
        y_pred_cross = self.one_step(X_cross, mag_scaler, model)
        multi_cross_loss = mean_squared_error(y_pred_cross[:, 0], mag_list_cross[self.window_len+1: -1])

        # # Test Prediction
        # y_pred_test = self.one_step(X_test, mag_scaler)

        multi_fit_fig = self.plot_prediction(t_list_train, mag_list_train, magerr_list_train,
                                             t_list_cross, mag_list_cross, magerr_list_cross,
                                             y_inter_train, y_pred_cross)

        multi_res_fig = self.plot_residual(t_list_train, mag_list_train, magerr_list_train,
                                           t_list_cross, mag_list_cross, magerr_list_cross,
                                           y_inter_train, y_pred_cross)

        return multi_train_loss, multi_cross_loss, multi_fit_fig, multi_res_fig

    def one_step(self, X, mag_scaler, model):
        # Cross Prediction
        scaled_y = []
        X_step = X[[0]]  # shape = (1, window_len, num_feature)
        print(np.shape(X))
        for i in range(np.shape(X)[0]):
            scaled_y_step = model.predict(X_step)  # shape = (1, num_feature)
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

