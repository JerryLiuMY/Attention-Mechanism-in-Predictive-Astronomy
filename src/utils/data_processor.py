import pickle
import math
import random
import numpy as np
import pandas as pd
import os
from global_setting import DATA_FOLDER
import json
import pickle
np.random.seed(1)


class DataProcessor:

    def __init__(self, crts_id):
        # Configuration
        self.crts_id = crts_id
        self.load_config()

        # Paths
        self.data_load_path = os.path.join(DATA_FOLDER, 'raw_data', str(self.crts_id) + '.csv')
        self.data_save_path = os.path.join(DATA_FOLDER, 'processed_data', str(self.crts_id) + '.pickle')

    def load_config(self):
        self.data_config = json.load(open('./config/data_config.json'))
        self.train_percent = self.data_config['data_loader']['train_partition']
        self.cross_percent = self.data_config['data_loader']['cross_partition']


    def partition_index(self, length):
        p1 = int(math.floor(length * self.train_percent))
        p2 = int(math.floor(length * (self.train_percent + self.cross_percent)))

        index_list = list(range(length))
        index_train = index_list[0:p1]
        index_cross = index_list[p1:p2]
        index_test = index_list[p2:]

        return index_train, index_cross, index_test

    def load_raw_data(self):
        with open(self.data_load_path) as handle:
            content = pd.read_csv(handle)
            mag_list_ = np.array(content['Mag'])
            magerr_list_ = np.array(content['Magerr'])
            mjd_list_ = np.array(content['MJD'])

            # Sort data
            index_list = np.argsort(mjd_list_)
            mag_list = np.expand_dims(mag_list_[index_list], axis=0)
            magerr_list = np.expand_dims(magerr_list_[index_list], axis=0)
            mjd_list = np.expand_dims(mjd_list_[index_list], axis=0)

            return mag_list, magerr_list, mjd_list


    def save_basic_data(self, mag_list, magerr_list, mjd_list, index_train, index_cross, index_test):
        t_list = mjd_list - mjd_list.min()

        mag_list_train = mag_list[index_train]
        mag_list_cross = mag_list[index_cross]
        mag_list_test = mag_list[index_test]
        magerr_list_train = magerr_list[index_train]
        magerr_list_cross = magerr_list[index_cross]
        magerr_list_test = magerr_list[index_test]
        t_list_train = t_list[index_train]
        t_list_cross = t_list[index_cross]
        t_list_test = t_list[index_test]

        data_dict = {'mag_list_train': mag_list_train, 'mag_list_cross': mag_list_cross, 'mag_list_test': mag_list_test,
                     'magerr_list_train': magerr_list_train, 'magerr_list_cross': magerr_list_cross,
                     'magerr_list_test': magerr_list_test, 't_list_train': t_list_train, 't_list_cross': t_list_cross,
                     't_list_test': t_list_test}

        with open(self.data_save_path, 'wb') as handle:
            pickle.dump(data_dict, handle)
