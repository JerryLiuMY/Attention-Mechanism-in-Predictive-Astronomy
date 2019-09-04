import math
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import json
from utils.data_loader import DataLoader
from operator import itemgetter



class GP():
    def __init__(self, data_dict):
        self.mag_list_train = data_dict['mag_list_train']
        self.mag_list_cross = data_dict['mag_list_cross']
        self.mag_list_test = data_dict['mag_list_test']
        self.magerr_list_train = data_dict['magerr_list_train']
        self.magerr_list_cross = data_dict['magerr_list_cross']
        self.magerr_list_test = data_dict['magerr_list_test']
        self.t_list_train = data_dict['t_list_train'].reshape(-1, 1)
        self.t_list_cross = data_dict['t_list_cross'].reshape(-1, 1)
        self.t_list_test = data_dict['t_list_test'].reshape(-1, 1)
        self.crts_id = data_dict['crts_id']
        self.data_config = json.load(open('./config/data_config.json'))
        self.model_config = json.load(open('./config/model_config.json'))

    @staticmethod
    def mse(y, y_pred):
        mse = np.mean((y - y_pred) ** 2)
        return mse

    def fit_gaussian(self):

        # Train gaussian process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        alpha = self.magerr_list_train ** 2
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)
        gp.fit(self.t_list_train, self.mag_list_train)  # MLE

        # Make predictions
        max_time = max(self.t_list_test)
        min_time = min(self.t_list_train)
        upper_limit = max_time + 0.2 * (max_time - min_time)
        lower_limit = min_time - 0.2 * (max_time - min_time)
        upper_limit = max_time
        lower_limit = min_time
        num = int(upper_limit - lower_limit)
        x = np.atleast_2d(np.linspace(lower_limit, upper_limit, num)).T
        y_pred, sigma = gp.predict(x, return_std=True)

        # Plot the function, the prediction and the 95% confidence interval based on
        # the MSE
        plt.figure(figsize=(12, 8))
        # plt.plot(x, self.f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
        plt.errorbar(self.t_list_train, self.mag_list_train, self.magerr_list_train,
                     fmt='r.', markersize=10, label='Training')
        plt.errorbar(self.t_list_cross, self.mag_list_cross, self.magerr_list_cross,
                     fmt='b.', markersize=10, label='Validation')
        plt.errorbar(self.t_list_test, self.mag_list_test, self.magerr_list_test,
                     fmt='g.', markersize=10, label='Test')
        plt.plot(x, y_pred, 'b-', label='Prediction')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.xlabel('MJD')
        plt.ylabel('Mag')
        plt.legend(loc='upper left')
        plt.title('crts_id: ' + str(self.crts_id))
        plt.show()