import pickle
import math
import random
import numpy as np
import pandas as pd
import os
from global_setting import DATA_FOLDER
import json
import pickle
import glob, os
np.random.seed(1)


class DataProcessor:

    def __init__(self):
        # Configuration
        self.load_config()
        self.load_crts_list()

        # Paths
        self.data_load_path = os.path.join(DATA_FOLDER, 'raw_data')
        self.data_save_path = os.path.join(DATA_FOLDER, 'processed_data', 'basic')

    def load_config(self):
        self.data_config = json.load(open('./config/data_config.json'))
        self.train_percent = self.data_config['data_loader']['train_partition']
        self.cross_percent = self.data_config['data_loader']['cross_partition']

    def load_crts_list(self):
        crts_list = []
        for file in os.listdir(os.path.join(DATA_FOLDER, 'raw_data')):
            if file.endswith(".csv"):
                crts_list.append(file.split('.')[0])
        self.crts_list = crts_list

    def partition_index(self, length):
        p1 = int(math.floor(length * self.train_percent))
        p2 = int(math.floor(length * (self.train_percent + self.cross_percent)))

        index_list = list(range(length))
        index_train = np.array(index_list[0:p1])
        index_cross = np.array(index_list[p1:p2])
        index_test = np.array(index_list[p2:])

        return index_train, index_cross, index_test

    def load_raw_data(self):
        mag_lists, magerr_lists, t_lists = [], [], []
        for crts_id in self.crts_list:
            with open(os.path.join(self.data_load_path, str(crts_id) + '.csv')) as handle:
                content = pd.read_csv(handle)
                mag_list_ = np.array(content['Mag'])
                magerr_list_ = np.array(content['Magerr'])
                mjd_list_ = np.array(content['MJD'])

                # Sort data
                index_list = np.argsort(mjd_list_)
                mag_list = mag_list_[index_list]
                magerr_list = magerr_list_[index_list]
                mjd_list = mjd_list_[index_list]
                t_list = mjd_list - mjd_list.min()

            mag_lists.append(mag_list)
            magerr_lists.append(magerr_list)
            t_lists.append(t_list)

        mag_lists = np.array(mag_lists)
        magerr_lists = np.array(magerr_lists)
        t_lists = np.array(t_lists)

        return mag_lists, magerr_lists, t_lists

    def save_basic_data(self, mag_lists, magerr_lists, t_lists):
        mag_lists_train, mag_lists_cross, mag_lists_test = [], [], []
        magerr_lists_train, magerr_lists_cross, magerr_lists_test = [], [], []
        t_lists_train, t_lists_cross, t_lists_test = [], [], []

        for i in range(len(mag_lists)):
            index_train, index_cross, index_test = self.partition_index(len(mag_lists[i]))
            mag_list_train = mag_lists[i][index_train]
            mag_list_cross = mag_lists[i][index_cross]
            mag_list_test = mag_lists[i][index_test]
            magerr_list_train = magerr_lists[i][index_train]
            magerr_list_cross = magerr_lists[i][index_cross]
            magerr_list_test = magerr_lists[i][index_test]
            t_list_train = t_lists[i][index_train]
            t_list_cross = t_lists[i][index_cross]
            t_list_test = t_lists[i][index_test]

            data_dict = {'mag_list_train': np.expand_dims(mag_list_cross, axis=0),
                         'mag_list_cross': np.expand_dims(mag_list_cross, axis=0),
                         'mag_list_test': np.expand_dims(mag_list_test, axis=0),
                         'magerr_list_train': np.expand_dims(magerr_list_train, axis=0),
                         'magerr_list_cross': np.expand_dims(magerr_list_cross, axis=0),
                         'magerr_list_test': np.expand_dims(magerr_list_test, axis=0),
                         't_list_train': np.expand_dims(t_list_train, axis=0),
                         't_list_cross': np.expand_dims(t_list_cross, axis=0),
                         't_list_test': np.expand_dims(t_list_test, axis=0)}

            # Save individual data
            with open(os.path.join(self.data_save_path, str(self.crts_list[i]) + '.pickle'), 'wb') as handle:
                pickle.dump(data_dict, handle)

            mag_lists_train.append(mag_list_train)
            mag_lists_cross.append(mag_list_cross)
            mag_lists_test.append(mag_list_test)
            magerr_lists_train.append(magerr_list_train)
            magerr_lists_cross.append(magerr_list_cross)
            magerr_lists_test.append(magerr_list_test)
            t_lists_train.append(t_list_train)
            t_lists_cross.append(t_list_cross)
            t_lists_test.append(t_list_test)

        mag_lists_train = np.array(mag_lists_train)
        mag_lists_cross = np.array(mag_lists_cross)
        mag_lists_train = np.array(mag_lists_train)
        magerr_lists_train = np.array(magerr_lists_train)
        magerr_lists_cross = np.array(magerr_lists_cross)
        magerr_lists_test = np.array(magerr_lists_test)
        t_lists_train = np.array(t_lists_train)
        t_lists_cross = np.array(t_lists_cross)
        t_lists_test = np.array(t_lists_test)

        all_data_dict = {'mag_lists_train': mag_lists_train, 'mag_lists_cross': mag_lists_cross,
                         'mag_lists_test': mag_lists_test, 'magerr_lists_train': magerr_lists_train,
                         'magerr_lists_cross': magerr_lists_cross, 'magerr_lists_test': magerr_lists_test,
                         't_lists_train': t_lists_train, 't_lists_cross': t_lists_cross, 't_lists_test': t_lists_test}

        # Save all data
        with open(os.path.join(self.data_save_path, 'all.pickle'), 'wb') as handle:
            pickle.dump(all_data_dict, handle)
