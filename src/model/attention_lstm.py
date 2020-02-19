import numpy as np
from keras.models import Model
import keras.backend as K
from keras.layers import Bidirectional, Input, LSTM
from keras.layers import RepeatVector, Concatenate, Dense, Dot, Activation
import matplotlib.pyplot as plt
from utils.phased_lstm import PhasedLSTM
from model.vanilla_lstm import VanillaLSTM
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam


def softmax(x, axis=1):
    # x = K.l2_normalize(x, axis=axis)
    # e = K.exp(x - K.max(x, axis=axis, keepdims=True))
    e = K.exp(x)
    s = K.sum(e, axis=axis, keepdims=True)
    return e / s


class AttentionLstm(VanillaLSTM):
    def __init__(self, window_len, hidden_dim, epochs, batch_size, phased):
        super().__init__(window_len, hidden_dim, epochs, batch_size, phased)
        # Configuration
        self.window_len = window_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.phased = phased
        self.model = None
        self.alpha_model = None

        self.repeator = RepeatVector(self.window_len)
        self.concatenator = Concatenate(axis=-1)
        self.densor = Dense(1, activation='tanh')
        self.activator = Activation(softmax, name='attention_weights')
        self.dotor = Dot(axes=1)

    def compute_attention(self, a, s_prev):
        s_prev = self.repeator(s_prev)
        concat = self.concatenator([a, s_prev])
        e = self.densor(concat)
        alpha = self.activator(e)
        context = self.dotor([alpha, a])

        return context, alpha

    def fit_model(self, X_train, y_train):
        X = Input(shape=(self.window_len, 2))
        s0 = Input(shape=(self.hidden_dim,), name='s0')
        c0 = Input(shape=(self.hidden_dim,), name='c0')

        if self.phased == 'phased':
            a = PhasedLSTM(self.hidden_dim, return_sequences=True)(X)
        else:
            a = LSTM(self.hidden_dim, return_sequences=True)(X)

        post_attention_LSTM_cell = LSTM(self.hidden_dim, return_state=True)
        output_layer = Dense(1, activation='linear')  # TODO: Try softmax activation (change input as well)

        outputs = []
        alphas = []
        s, c = s0, c0
        # Step 2.2: Iterate for Ty steps
        for t in range(1):
            context, alpha = self.compute_attention(a, s)
            s, _, c = post_attention_LSTM_cell(context, initial_state=[s, c])
            output = output_layer(s)
            outputs.append(output)
            alphas.append(alpha)

        # Step 3: Create model instance taking three inputs and returning the list of outputs
        self.model = Model(inputs=[X, s0, c0], outputs=outputs)
        self.model.summary()
        self.alpha_model = Model(inputs=[X, s0, c0], outputs=alphas)

        s0_train = np.zeros((np.shape(X_train)[0], self.hidden_dim))
        c0_train = np.zeros((np.shape(X_train)[0], self.hidden_dim))
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
        self.model.compile(loss='mean_squared_error', optimizer=adam)
        self.model.fit([X_train, s0_train, c0_train], y_train, epochs=self.epochs, batch_size=self.batch_size)

    def run_model(self, X, **kwargs):
        s0 = kwargs['s0']
        c0 = kwargs['c0']
        scaled_y_pred = self.model.predict(X, s0, c0)
        return scaled_y_pred

    def single_step_prediction(self, t_train, mag_train, magerr_train,
                               t_cross, mag_cross, magerr_cross,
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
        # attention_weight = np.zeros([50, 50])
        attention_weight = self.alpha_model.predict([X_train, s0_train, c0_train])
        y_inter_train = mag_scaler.inverse_transform(scaled_y_inter)
        single_train_loss = mean_squared_error(y_inter_train[:, 0], mag_train[self.window_len + 1: -1])

        # Cross Prediction
        scaled_cross_y_pred = self.model.predict([X_cross, s0_cross, c0_cross])
        y_pred_cross = mag_scaler.inverse_transform(scaled_cross_y_pred)
        single_cross_loss = mean_squared_error(y_pred_cross[:, 0], mag_cross[self.window_len + 1: -1])

        # # Test Prediction
        # scaled_test_y_pred = self.model.predict([X_test, s0_test, c0_test])
        # y_pred_test = mag_scaler.inverse_transform(scaled_test_y_pred)
        # single_test_loss = mean_squared_error(y_pred_test[:, 0], magerr_list_test[self.window_len + 1: -1])

        single_fit_fig = self.plot_prediction(t_train, mag_train, magerr_train,
                                              t_cross, mag_cross, magerr_cross,
                                              y_inter_train, y_pred_cross)

        single_res_fig = self.plot_residual(t_train, mag_train, magerr_train,
                                            t_cross, mag_cross, magerr_cross,
                                            y_inter_train, y_pred_cross)

        attention_fig = self.attention_visualization(attention_weight)

        return single_train_loss, single_cross_loss, attention_fig, single_fit_fig, single_res_fig

    def multi_step_prediction(self, t_train, mag_train, magerr_train,
                              t_cross, mag_cross, magerr_cross,
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
        y_inter_train = mag_scaler.inverse_transform(scaled_y_inter)
        multi_train_loss = mean_squared_error(y_inter_train[:, 0], mag_train[self.window_len + 1: -1])

        # Cross Prediction
        y_pred_cross = self.one_step(X_cross, s0_cross, c0_cross, mag_scaler)
        multi_cross_loss = mean_squared_error(y_pred_cross[:, 0], mag_cross[self.window_len + 1: -1])

        # # Test Prediction
        # y_pred_test = self.one_step(X_test, s0_test, c0_test, mag_scaler)

        multi_fit_fig = self.plot_prediction(t_train, mag_train, magerr_train,
                                             t_cross, mag_cross, magerr_cross,
                                             y_inter_train, y_pred_cross)

        multi_res_fig = self.plot_residual(t_train, mag_train, magerr_train,
                                           t_cross, mag_cross, magerr_cross,
                                           y_inter_train, y_pred_cross)

        return multi_train_loss, multi_cross_loss, multi_fit_fig, multi_res_fig

    def one_step(self, X, mag_scaler, **kwargs):
        s0 = kwargs['s0']
        c0 = kwargs['c0']
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

    @staticmethod
    def attention_visualization(attention_weight):
        print('attention')
        print(np.shape(attention_weight))
        attention_weight = attention_weight[:, :, 0]
        attention_fig = plt.figure(figsize=(12, 8))
        plt.matshow(attention_weight)
        plt.colorbar()
        plt.show()
        plt.xlabel('look back')
        plt.ylabel('sample')

        return attention_fig
