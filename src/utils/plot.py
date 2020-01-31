import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from global_setting import DATA_FOLDER
from sklearn.metrics import mean_squared_error


def plot_series():
    plt.figure(figsize=(15, 15))
    i = 0
    for file in glob.glob(os.path.join(DATA_FOLDER, '*.csv')):
        i += 1
        with open(file) as handle:
            content = pd.read_csv(handle)
            mag_list = np.array(content['Mag'])
            mjd = np.array(content['MJD'])
            plt.subplot(5, 2, i)
            plt.scatter(mjd, mag_list)
            plt.xlabel('MJD')
            plt.ylabel('Mag')
            plt.title(file.partition(file)[0])

    plt.show()


def plot_series_short():
    plt.figure(figsize=(15, 15))
    i = 0
    for file in glob.glob(os.path.join(DATA_FOLDER, '*.csv')):
        i += 1
        with open(file) as handle:
            content = pd.read_csv(handle)
            mag_list = np.array(content['Mag'])[0:30]
            magerr_list = np.array(content['Magerr'])[0:30]
            mjd = np.array(content['MJD'])[0:30]
            plt.subplot(5, 2, i)
            plt.errorbar(mjd, mag_list, yerr=magerr_list, fmt='o')
            plt.xlabel('MJD')
            plt.ylabel('Mag')
            plt.title(file.partition(file)[0])

    plt.show()


def fill_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train, y_pred_var_train,
              t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, y_pred_var_cross):

    fill_plot = plt.figure(figsize=(12, 8))
    plt.errorbar(t_train, mag_train, yerr=magerr_train, fmt='k.')
    plt.errorbar(t_train, mag_train, yerr=magerr_cross, fmt='k.')

    # Interpolation
    train_loss = mean_squared_error(y_pred_train, mag_train)
    plt.plot(t_pred_train, y_pred_train, color='green', ls='-')
    plt.fill_between(t_pred_train, y1=y_pred_train + np.sqrt(y_pred_var_train),
                     y2=y_pred_train - np.sqrt(y_pred_var_train), color='LimeGreen', alpha=0.5)

    # Prediction
    cross_loss = mean_squared_error(y_pred_cross, mag_cross)
    plt.plot(t_pred_cross, y_pred_cross, color='blue', ls='-')
    plt.fill_between(t_pred_cross, y1=y_pred_cross + np.sqrt(y_pred_var_cross),
                     y2=y_pred_cross - np.sqrt(y_pred_var_cross), color='DodgerBlue', alpha=0.5)

    # Decoration
    plt.xlim(t_train.min(), t_cross.max())
    plt.xlabel('Time[days]')
    plt.ylabel('Magnitude')
    plt.title('Simulated Average Paths')

    return train_loss, cross_loss, fill_plot


def scatter_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train,
                 t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, type):

    # Plot the function, the prediction and the 95% confidence interval based on the MSE
    scatter_plot = plt.figure(1, figsize=(12, 8))
    plt.errorbar(t_train, mag_train, magerr_train, fmt='k.', markersize=10, label='Training')
    plt.errorbar(t_cross, mag_cross, magerr_cross, fmt='k.', markersize=10, label='Validation')
    plt.scatter(t_pred_train, y_pred_train, color='g')
    plt.scatter(t_pred_cross, y_pred_cross, color='b')

    # Decoration
    plt.xlim(t_train.min(), t_cross.max())
    plt.xlabel('MJD')
    plt.ylabel('Mag')
    plt.legend(loc='upper left')
    plt.title(type)
    plt.show()

    return scatter_plot


    residual_pred_train = y_pred_train[:, 0] - mag_train[window_len + 1: -1]
    residual_pred_cross = y_pred_cross[:, 0] - mag_cross[window_len + 1: -1]
    zeros_train = np.zeros(len(magerr_train))
    zeros_cross = np.zeros(len(magerr_cross))