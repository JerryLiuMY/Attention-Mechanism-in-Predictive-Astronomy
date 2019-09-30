from utils.phased_lstm_layer import PhasedLSTMCell, PhasedLSTM
from keras.layers import Dense
from global_setting import DATA_FOLDER
from global_setting import phased_lstm_model_path
import numpy as np
import os
import json
from keras.models import Sequential, load_model

class PhasedLSTM_:
    # The model is now trained individually for each sample, so we feed in the crts_id for now
    def __init__(self, crts_id):
        # Configuration
        self.crts_id = crts_id
        self.model = None
        self.load_config()

        # Model Path & Name
        self.phased_lstm_model_name = str(self.crts_id) + '_window_len_' + str(self.window_len) + '_hidden_dim_' + str(self.hidden_dim) + '.h5'
        self.phased_lstm_model_path = phased_lstm_model_path

    def load_config(self):
        self.model_config = json.load(open('./config/model_config.json'))
        self.epochs = self.model_config['phased_lstm']['epochs']
        self.batch_size = self.model_config['phased_lstm']['batch_size']
        self.hidden_dim = self.model_config['phased_lstm']['hidden_dim']
        self.window_len = self.model_config['phased_lstm']['window_len']

    def build_model(self):
        # Build Model
        self.model = Sequential()
        self.model.add(PhasedLSTM(self.hidden_dim, input_shape=(self.window_len, 2)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit_model(self, train_X, train_y, cross_X, cross_y, test_X, test_y):
        # Train Model
        train_score = self.model.fit(train_X, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        corss_score = self.model.evaluate(cross_X, cross_y, batch_size=self.batch_size)
        test_score = self.model.evaluate(test_X, test_y, batch_size=self.batch_size)

        # Save model
        self.model.save(os.path.join(self.phased_lstm_model_path, self.phased_lstm_model_name))

        return train_score, corss_score, test_score

    def one_step_prediction(self, train_X, cross_X, test_X, mag_scaler):
        # Load Model
        phased_lstm_model = load_model(os.path.join(self.phased_lstm_model_path, self.phased_lstm_model_name))

        # Train Interpolation
        scaled_y_inter = phased_lstm_model.predict(train_X)
        y_inter = mag_scaler.inverse_transform(scaled_y_inter)

        # Cross Prediction
        scaled_cross_y_pred = phased_lstm_model.predict(cross_X)
        cross_y_pred = mag_scaler.inverse_transform(scaled_cross_y_pred)

        # Test Prediction
        scaled_test_y_pred = phased_lstm_model.predict(test_X)
        test_y_pred = mag_scaler.inverse_transform(scaled_test_y_pred)

        return y_inter, cross_y_pred, test_y_pred

    def multiple_step_prediction(self, train_X, cross_X, test_X, mag_scaler):
        # Load Model
        phased_lstm_model = load_model(os.path.join(self.phased_lstm_model_path, self.phased_lstm_model_name))

        # Train Interpolation
        scaled_y_inter = phased_lstm_model.predict(train_X)
        y_inter = mag_scaler.inverse_transform(scaled_y_inter)

        # Cross Prediction
        cross_X_step = cross_X[[0]]
        scaled_y_pred_cross = []
        for i in range(np.shape(cross_X)[0]):
            scaled_cross_y_pred_step = phased_lstm_model.predict(cross_X_step)  # shape = (1, 1)
            scaled_cross_t_step = cross_X[i+1][[-1], :-1]  # shape = (1, 1)
            new_feature = np.concatenate((scaled_cross_y_pred_step, scaled_cross_t_step), axis=1)  # shape = (1, 2)
            cross_X_step = np.concatenate((cross_X[i][1:, :], new_feature), axis=0)  # shape = (window_len, 2)
            cross_X_step = np.expand_dims(cross_X_step, axis=0)  # shape = (1, window_len, 2)

            scaled_y_pred_cross.append(scaled_cross_y_pred_step[0])

        scaled_y_pred_cross = np.array(scaled_y_pred_cross)
        y_pred_cross = mag_scaler.inverse_transform(scaled_y_pred_cross)

        # Cross Prediction
        test_X_step = test_X[[0]]
        scaled_y_pred_test = []
        for i in range(np.shape(test_X)[0]):
            scaled_test_y_pred_step = phased_lstm_model.predict(test_X_step)  # shape = (1, 1)
            scaled_test_t_step = test_X[i+1][[-1], :-1]  # shape = (1, 1)
            new_feature = np.concatenate((scaled_test_y_pred_step, scaled_test_t_step), axis=1)  # shape = (1, 2)
            test_X_step = np.concatenate((test_X[i][1:, :], new_feature), axis=0)  # shape = (window_len, 2)
            test_X_step = np.expand_dims(test_X_step, axis=0)  # shape = (1, window_len, 2)

            scaled_y_pred_test.append(scaled_test_y_pred_step[0])

        scaled_y_pred_test = np.array(scaled_y_pred_test)
        y_pred_test = mag_scaler.inverse_transform(scaled_y_pred_test)

        return y_inter, y_pred_cross, y_pred_test