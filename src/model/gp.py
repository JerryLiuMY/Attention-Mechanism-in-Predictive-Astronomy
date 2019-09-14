import math
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import json
import os
import pickle
from global_setting import DATA_FOLDER


class GP():
    def __init__(self, crts_id):
        self.crts_id = crts_id
        self.data_path = os.path.join(DATA_FOLDER, 'processed_data', 'basic', str(crts_id) + '.pickle')
        with open('self.data_path', 'rb') as handle:
            data_dict = pickle.load(self.data_path)
        self.mag_list_train = data_dict['mag_list_train']
        self.mag_list_cross = data_dict['mag_list_cross']
        self.mag_list_test = data_dict['mag_list_test']
        self.magerr_list_train = data_dict['magerr_list_train']
        self.magerr_list_cross = data_dict['magerr_list_cross']
        self.magerr_list_test = data_dict['magerr_list_test']
        self.t_list_train = data_dict['t_list_train']
        self.t_list_cross = data_dict['t_list_cross']
        self.t_list_test = data_dict['t_list_test']
        self.data_config = json.load(open('./config/data_config.json'))
        self.model_config = json.load(open('./config/model_config.json'))
        self.gp = None

    def fit_gaussian(self):
        # Train gaussian process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        alpha = self.magerr_list_train ** 2
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)
        self.gp.fit(self.t_list_train, self.mag_list_train-18.15)  # MLE

    def make_prediction(self):
        # Make predictions
        max_time = self.t_list_test.max()
        min_time = self.t_list_train.min()
        upper_limit = max_time
        lower_limit = min_time
        num = int((upper_limit - lower_limit)/0.2)
        x = np.atleast_2d(np.linspace(lower_limit, upper_limit, num)).T
        y_pred, sigma = self.gp.predict(x, return_std=True)
        y_pred = y_pred+18.15

        # Plot the function, the prediction and the 95% confidence interval based on the MSE
        plt.figure(figsize=(12, 8))
        plt.errorbar(self.t_list_train, self.mag_list_train, self.magerr_list_train,
                     fmt='g.', markersize=10, label='Training')
        plt.errorbar(self.t_list_cross, self.mag_list_cross, self.magerr_list_cross,
                     fmt='b.', markersize=10, label='Validation')
        plt.errorbar(self.t_list_test, self.mag_list_test, self.magerr_list_test,
                     fmt='r.', markersize=10, label='Test')
        plt.plot(x, y_pred, 'b-', label='Prediction')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
                 alpha=.5, fc='DodgerBlue', ec='None', label='95% confidence interval')
        plt.xlim(lower_limit, self.t_list_cross.max()-10)
        plt.xlabel('MJD')
        plt.ylabel('Mag')
        plt.legend(loc='upper left')
        plt.title('crts_id: ' + str(self.crts_id))
        plt.show()