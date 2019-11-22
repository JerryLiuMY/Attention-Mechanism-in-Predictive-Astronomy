import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Bidirectional, Input, LSTM, Softmax
from keras.layers import RepeatVector, Concatenate, Dense, Dot, Activation, Lambda
from utils.phased_lstm import PhasedLSTM
from keras.models import Model
from model.c_vanilla_lstm import VanillaLSTM
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam


class AttentionLstm(VanillaLSTM):
    def __init__(self, window_len, hidden_dim, epochs, batch_size, phased='phased'):
        super().__init__(window_len, hidden_dim, epochs, batch_size, phased)
        # Configuration
        self.window_len = window_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.phased = phased
        self.model = None

        self.repeator = RepeatVector(self.window_len)
        self.concatenator = Concatenate(axis=-1)
        self.densor = Dense(1, activation="relu")
        self.activator = Softmax(axis=-1)
        self.dotor = Dot(axes=1)

    def compute_attention(self, a, s_prev):
        s_prev = self.repeator(s_prev)
        concat = self.concatenator([a, s_prev])
        e = self.densor(concat)  # e: scalar - un-normalized attention weight
        alphas = self.activator(e)  # alphas: scalar - normalized attention weight
        context = self.dotor([alphas, a])

        return context

    def build_model(self):
        # Step 1.1: Input
        X = Input(shape=(self.window_len, 2))
        s0 = Input(shape=(self.hidden_dim,), name='s0')
        c0 = Input(shape=(self.hidden_dim,), name='c0')

        if self.phased == 'phased':
            # Step 1.2: Pre-attention Bidirectional LSTM
            a = Bidirectional(PhasedLSTM(self.hidden_dim, return_sequences=True))(X)

            # Step 1.3: Post-attention Bidirectional LSTM
            post_attention_LSTM_cell = LSTM(self.hidden_dim, return_state=True)

        else:
            # Step 1.2: Pre-attention Bidirectional LSTM
            a = Bidirectional(LSTM(self.hidden_dim, return_sequences=True))(X)

            # Step 1.3: Post-attention Bidirectional LSTM
            post_attention_LSTM_cell = PhasedLSTM(self.hidden_dim, return_state=True)

        # Step 1.4: Output
        output_layer = Dense(1)

        # Step 2.1: Initialize
        outputs = []
        s = s0
        c = c0

        # Step 2.2: Iterate for Ty steps
        for t in range(1):
            context = self.compute_attention(a, s)
            s, _, c = post_attention_LSTM_cell(context, initial_state=[s, c])
            output = output_layer(s)
            outputs.append(output)

        # Step 3: Create model instance taking three inputs and returning the list of outputs
        self.model = Model(inputs=[X, s0, c0], outputs=outputs)
        self.model.summary()


    def fit_model(self, X_train, y_train, X_cross, y_cross, X_test, y_test):
        # Initialize State
        s0_train = np.zeros((np.shape(X_train)[0], self.hidden_dim))
        c0_train = np.zeros((np.shape(X_train)[0], self.hidden_dim))
        s0_cross = np.zeros((np.shape(X_cross)[0], self.hidden_dim))
        c0_cross = np.zeros((np.shape(X_cross)[0], self.hidden_dim))
        s0_test = np.zeros((np.shape(X_test)[0], self.hidden_dim))
        c0_test = np.zeros((np.shape(X_test)[0], self.hidden_dim))

        # adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
        # adam = Adam(lr=0.01)
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit([X_train, s0_train, c0_train], y_train, epochs=self.epochs, batch_size=self.batch_size)
        self.model.evaluate([X_cross, s0_cross, c0_cross], y_cross, batch_size=self.batch_size)
        self.model.evaluate([X_test, s0_test, c0_test], y_test, batch_size=self.batch_size)

        return self.model

    def single_step_prediction(self, t_list_train, mag_list_train, magerr_list_train,
                               t_list_cross, mag_list_cross, magerr_list_cross,
                               X_train, X_cross, X_test, mag_scaler):
        # Initialize State
        s0_train = np.zeros((np.shape(X_train)[0], self.hidden_dim))
        c0_train = np.zeros((np.shape(X_train)[0], self.hidden_dim))
        s0_cross = np.zeros((np.shape(X_cross)[0], self.hidden_dim))
        c0_cross = np.zeros((np.shape(X_cross)[0], self.hidden_dim))
        s0_test = np.zeros((np.shape(X_test)[0], self.hidden_dim))
        c0_test = np.zeros((np.shape(X_test)[0], self.hidden_dim))

        # Train Interpolation
        scaled_y_inter = self.model.predict([X_train, s0_train, c0_train])
        y_inter = mag_scaler.inverse_transform(scaled_y_inter)

        # Cross Prediction
        scaled_cross_y_pred = self.model.predict([X_cross, s0_cross, c0_cross])
        y_pred_cross = mag_scaler.inverse_transform(scaled_cross_y_pred)
        single_cross_loss = mean_squared_error(y_pred_cross[:, 0], mag_list_cross[self.window_len + 1: -1])

        # # Test Prediction
        # scaled_test_y_pred = self.model.predict([X_test, s0_test, c0_test])
        # y_pred_test = mag_scaler.inverse_transform(scaled_test_y_pred)

        fit_fig = self.plot_prediction(t_list_train, mag_list_train, magerr_list_train,
                                       t_list_cross, mag_list_cross, magerr_list_cross,
                                       y_inter, y_pred_cross)

        res_fig = self.plot_residual(t_list_train, mag_list_train, magerr_list_train,
                                     t_list_cross, mag_list_cross, magerr_list_cross,
                                     y_inter, y_pred_cross)

        return single_cross_loss, fit_fig, res_fig

    def multi_step_prediction(self, t_list_train, mag_list_train, magerr_list_train,
                              t_list_cross, mag_list_cross, magerr_list_cross,
                              X_train, X_cross, X_test, mag_scaler):
        # Initialize State
        s0_train = np.zeros((np.shape(X_train)[0], self.hidden_dim))
        c0_train = np.zeros((np.shape(X_train)[0], self.hidden_dim))
        s0_cross = np.zeros((np.shape(X_cross)[0], self.hidden_dim))
        c0_cross = np.zeros((np.shape(X_cross)[0], self.hidden_dim))
        s0_test = np.zeros((np.shape(X_test)[0], self.hidden_dim))
        c0_test = np.zeros((np.shape(X_test)[0], self.hidden_dim))

        # Train Interpolation
        scaled_y_inter = self.model.predict([X_train, s0_train, c0_train])
        y_inter = mag_scaler.inverse_transform(scaled_y_inter)

        # Cross Prediction
        y_pred_cross = self.one_step(X_cross, s0_cross, c0_cross, mag_scaler)
        multi_cross_loss = mean_squared_error(y_pred_cross[:, 0], mag_list_cross[self.window_len+1: -1])

        # # Test Prediction
        # y_pred_test = self.one_step(X_test, s0_test, c0_test, mag_scaler)

        fit_fig = self.plot_prediction(t_list_train, mag_list_train, magerr_list_train,
                                       t_list_cross, mag_list_cross, magerr_list_cross,
                                       y_inter, y_pred_cross)

        res_fig = self.plot_residual(t_list_train, mag_list_train, magerr_list_train,
                                     t_list_cross, mag_list_cross, magerr_list_cross,
                                     y_inter, y_pred_cross)

        return multi_cross_loss, fit_fig, res_fig

    def one_step(self, X, s0, c0, mag_scaler):
        # Cross Prediction
        scaled_y = []
        X_step = X[[0]]  # shape = (1, window_len, num_feature)
        print(np.shape(X))
        for i in range(np.shape(X)[0]):
            scaled_y_step = self.model.predict([X_step, s0, c0])  # shape = (1, num_feature)
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