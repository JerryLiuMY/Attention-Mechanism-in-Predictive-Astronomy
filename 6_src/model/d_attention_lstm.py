import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Bidirectional, Input, LSTM, Softmax
from keras.layers import RepeatVector, Concatenate, Dense, Dot, Activation, Lambda
from keras.models import Model
from keras.optimizers import Adam


class AttentionLstm():

    def __init__(self, window_len, hidden_dim, epochs, batch_size):
        # Configuration
        self.window_len = window_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None


    def compute_attention(self, a, s_prev):
        repeator = RepeatVector(self.window_len)
        s_prev = repeator(s_prev)

        concatenator = Concatenate(axis=-1)
        concat = concatenator([a, s_prev])

        densor = Dense(1, activation="relu")
        e = densor(concat) # e: scalar - un-normalized attention weight

        activator = Softmax(axis=-1)
        alphas = activator(e)  # alphas: scalar - normalized attention weight

        dotor = Dot(axes=1)
        context = dotor([alphas, a])

        return context

    def build_model(self):
        # Step 1.1: Input
        X = Input(shape=(self.window_len, 2))
        s0 = Input(shape=(self.hidden_dim,), name='s0')
        c0 = Input(shape=(self.hidden_dim,), name='c0')

        # Step 1.2: Pre-attention Bidirectional LSTM
        a = Bidirectional(LSTM(self.hidden_dim, return_sequences=True))(X)

        # Step 1.3: Post-attention Bidirectional LSTM
        post_attention_LSTM_cell = LSTM(self.hidden_dim, return_state=True)

        # Step 1.4: Output
        output_layer = Dense(1, activation='softmax')

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


    def fit_model(self, train_X, train_y, cross_X, cross_y, test_X, test_y):
        s0 = np.zeros((np.shape(train_X)[0], self.hidden_dim))
        c0 = np.zeros((np.shape(train_X)[0], self.hidden_dim))
        opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        train_score = self.model.fit([train_X, s0, c0], train_y, epochs=self.epochs, batch_size=self.batch_size)
        corss_score = self.model.evaluate(cross_X, cross_y, batch_size=self.batch_size)
        test_score = self.model.evaluate(test_X, test_y, batch_size=self.batch_size)

        return self.model


    def multi_step_prediction(self, t_list_train, mag_list_train, magerr_list_train,
                              t_list_cross, mag_list_cross, magerr_list_cross,
                              X_train, X_cross, X_test, mag_scaler):
        # Train Interpolation
        scaled_y_inter = self.model.predict(X_train)
        y_inter = mag_scaler.inverse_transform(scaled_y_inter)

        # Cross Prediction
        scaled_cross_y_pred = self.model.predict(X_cross)
        y_pred_cross = mag_scaler.inverse_transform(scaled_cross_y_pred)

        # # Test Prediction
        # scaled_test_y_pred = self.model.predict(X_test)
        # y_pred_test = mag_scaler.inverse_transform(scaled_test_y_pred)

        fig = self.plot_prediction(t_list_train, mag_list_train, magerr_list_train,
                                   t_list_cross, mag_list_cross, magerr_list_cross,
                                   y_inter, y_pred_cross)

        return fig

    def plot_prediction(self, t_list_train, mag_list_train, magerr_list_train,
                        t_list_cross, mag_list_cross, magerr_list_cross,
                        y_inter, y_pred_cross):

        # Specify parameters
        max_time = t_list_cross.max()
        min_time = t_list_train.min()
        upper_limit = max_time
        lower_limit = min_time

        # Plot the function, the prediction and the 95% confidence interval based on the MSE
        fig = plt.figure(figsize=(12, 8))
        plt.errorbar(t_list_train, mag_list_train, magerr_list_train,
                     fmt='k.', markersize=10, label='Training')
        plt.errorbar(t_list_cross, mag_list_cross, magerr_list_cross,
                     fmt='k.', markersize=10, label='Validation')
        plt.scatter(t_list_train[self.window_len+1: -1], y_inter, color='g')
        plt.scatter(t_list_cross[self.window_len+1: -1], y_pred_cross, color='b')
        plt.xlim(lower_limit, upper_limit)
        plt.xlabel('MJD')
        plt.ylabel('Mag')
        plt.legend(loc='upper left')
        plt.title('Simulated Path')
        plt.show()

        return fig