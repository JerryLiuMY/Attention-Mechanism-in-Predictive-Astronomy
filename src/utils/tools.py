import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from global_setting import DATA_FOLDER, WINDOW_LEN


def discrete_decorator(original_function):
    def wrapper(*args, **kwargs):
        sim_fig = original_function(*args, **kwargs, fig_type='simulated')
        res_fig = original_function(*args, **kwargs, fig_type='residual')
        return sim_fig, res_fig
    return wrapper


def continuous_decorator(original_function):
    def wrapper(*args, **kwargs):
        sample_fig = original_function(*args, **kwargs, fig_type='sample')
        average_fig = original_function(*args, **kwargs, fig_type='average')
        return sample_fig, average_fig
    return wrapper


@discrete_decorator
def discrete_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train, y_pred_var_train,
                  t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, y_pred_var_cross,
                  fig_type):
    discrete_fig = plt.figure(1, figsize=(12, 8))

    if fig_type == 'simulated':
        plt.errorbar(t_train, mag_train, magerr_train, fmt='k.', markersize=10, label='Training')
        plt.errorbar(t_cross, mag_cross, magerr_cross, fmt='k.', markersize=10, label='Validation')
        plt.scatter(t_pred_train, y_pred_train, color='g')
        plt.scatter(t_pred_cross, y_pred_cross, color='b')

    elif fig_type == 'residual':
        plt.errorbar(t_train, np.zeros(len(mag_train)), magerr_train, fmt='k.', markersize=10, label='Training')
        plt.errorbar(t_cross, np.zeros(len(mag_train)), magerr_cross, fmt='k.', markersize=10, label='Validation')
        plt.scatter(t_pred_train, y_pred_train[:, 0] - mag_train[WINDOW_LEN+1: -1], color='g')
        plt.scatter(t_pred_cross, y_pred_cross[:, 0] - mag_cross[WINDOW_LEN+1: -1], color='b')

    else:
        raise Exception('Not valid fig_type')

    # Decoration
    plt.xlim(t_train.min(), t_cross.max())
    plt.xlabel('MJD')
    plt.ylabel('Mag')
    plt.title(' '.join([fig_type.capitalize(), 'Plot']))

    return discrete_fig


@continuous_decorator
def continuous_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train, y_pred_var_train, y_pred_train_n,
                    t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, y_pred_var_cross, y_pred_cross_n,
                    fig_type):
    continuous_fig = plt.figure(figsize=(12, 8))
    plt.errorbar(t_train, mag_train, yerr=magerr_train, fmt='k.')
    plt.errorbar(t_train, mag_cross, yerr=magerr_cross, fmt='k.')

    if fig_type == 'sample':
        n_paths = np.shape(y_pred_train_n)[0]
        for i in range(n_paths):
            plt.plot(t_pred_train, y_pred_train_n[i], color='green', alpha=(1 - i / float(n_paths)))
            plt.plot(t_pred_cross, y_pred_cross_n[i], color='blue', alpha=(1 - i / float(n_paths)))

    elif fig_type == 'average':
        plt.plot(t_pred_train, y_pred_train, color='green', ls='-')
        plt.plot(t_pred_cross, y_pred_cross, color='blue', ls='-')
        plt.fill_between(t_pred_train, y1=y_pred_train + np.sqrt(y_pred_var_train),
                         y2=y_pred_train - np.sqrt(y_pred_var_train), color='LimeGreen', alpha=0.5)
        plt.fill_between(t_pred_cross, y1=y_pred_cross + np.sqrt(y_pred_var_cross),
                         y2=y_pred_cross - np.sqrt(y_pred_var_cross), color='DodgerBlue', alpha=0.5)

    else:
        raise Exception('Not valid fig_type')

    # Decoration
    plt.xlim(t_train.min(), t_cross.max())
    plt.xlabel('Time[days]')
    plt.ylabel('Magnitude')
    plt.title(' '.join(['Simulated', fig_type.capitalize(), 'Paths']))

    return continuous_fig


def match_list(t, t_pred, y_pred, y_pred_var):
    y_pred_match = []
    y_pred_var_match = []
    for date in t:
        t_id = min(range(len(t_pred)), key=lambda i: abs(t_pred[i] - date))
        y_pred_match.append(y_pred[t_id])
        y_pred_var_match.append(y_pred_var[t_id])

    return y_pred_match, y_pred_var_match


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
