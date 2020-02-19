import numpy as np
import pandas as pd
import os
import json
import pickle
import joblib
from keras.models import load_model
from utils.data_processor import BasicDataProcessor, LSTMDataProcessor
from global_setting import DATA_FOLDER, crts_list
from global_setting import raw_data_folder, basic_data_folder, lstm_data_folder
from global_setting import model_folder, figure_folder
from global_setting import result_csv

# from model.a_carma import Carma
from model.gp import GP
from model.vanilla_lstm import VanillaLSTM
from model.attention_lstm import AttentionLstm
from model.bayesian_lstm import BayesianLSTM
from utils.phased_lstm import PhasedLSTM


model_type = ['carma', 'gp', 'vanilla', 'attention']
phased_type = ['standard', 'phased']
dc_type = ['discrete', 'continuous']
sm_type = ['single', 'multiple']
set_type = ['train', 'cross']

columns = []
for model in model_type:
    for phased in phased_type:
        for dc in dc_type:
            for sm in sm_type:
                for set_name in set_type:
                    column_name = '_'.join([model, phased, dc, sm, set_name])
                    columns.append(column_name)

def initialize_csv():
    result_df = pd.DataFrame(columns=columns)
    result_df.to_csv(result_csv)


class MainPipeline:
    def __init__(self, crts_id):
        # Configuration
        self.crts_id = crts_id

        # Data Folder
        self.raw_data_folder = raw_data_folder
        self.basic_data_folder = basic_data_folder
        self.carma_data_folder = raw_data_folder
        self.lstm_data_folder = lstm_data_folder

        # Model Folder
        self.model_folder = model_folder
        self.figure_folder = figure_folder
        self.load_name()
        self.load_model_config()
        self.prepare_individual_data()
        self.load_individual_data()

    def load_name(self):
        self.basic_data_name = self.crts_id + '.pkl'
        self.rescaled_mag_name = '_'.join([self.crts_id, 'rescaled_mag.pkl'])
        self.mag_scaler_name = '_'.join([self.crts_id, 'mag_scaler.pkl'])
        self.rescaled_delta_t_name = '_'.join([self.crts_id, 'rescaled_delta_t.pkl'])
        self.delta_t_scaler_name = '_'.join([self.crts_id, 'delta_t_scaler.pkl'])
        self.standard_X_y_name = '_'.join([self.crts_id, 'X_y.plk'])

    def load_model_config(self):
        self.model_config = json.load(open('./config/model_config.json'))
        self.carma_config = self.model_config["carma"]
        self.lstm_config = self.model_config["lstm"]

    def prepare_individual_data(self):
        basic_data_processor = BasicDataProcessor(self.crts_id)
        basic_data_processor.prepare_basic_data()
        lstm_data_processor = LSTMDataProcessor(self.crts_id)
        lstm_data_processor.prepare_lstm_data()

    def load_individual_data(self):
        # Basic Data
        basic_data_path = os.path.join(self.basic_data_folder, self.basic_data_name)
        with open(basic_data_path, 'rb') as handle:
            data_dict = pickle.load(handle)
        self.mag_train = data_dict['mag_train']
        self.mag_cross = data_dict['mag_cross']
        self.mag_test = data_dict['mag_test']
        self.magerr_train = data_dict['magerr_train']
        self.magerr_cross = data_dict['magerr_cross']
        self.magerr_test = data_dict['magerr_test']
        self.t_train = data_dict['t_train']
        self.t_cross = data_dict['t_cross']
        self.t_test = data_dict['t_test']

        # LSTM Data
        self.mag_scaler = joblib.load(os.path.join(self.lstm_data_folder, self.mag_scaler_name))
        self.delta_t_scaler = joblib.load(os.path.join(self.lstm_data_folder, self.delta_t_scaler_name))
        with open(os.path.join(self.lstm_data_folder, self.standard_X_y_name), 'rb') as handle:
            X_y_data_dict = pickle.load(handle)
            self.X_train = X_y_data_dict['X_train']
            self.y_train = X_y_data_dict['y_train']
            self.X_cross = X_y_data_dict['X_cross']
            self.y_cross = X_y_data_dict['y_cross']
            self.X_test = X_y_data_dict['X_test']
            self.y_test = X_y_data_dict['y_test']

    # run model
    def run_lstm(self, model_type, phased_type, dc_type, sm_type, n_walkers, train=True):
        # Model config
        result_df = pd.read_csv(result_csv, index_col=0, encoding='utf-8')
        epochs = self.lstm_config["epochs"]
        batch_size = self.lstm_config["batch_size"]
        hidden_dim = self.lstm_config["hidden_dim"]
        model_map = {'vanilla_lstm': VanillaLSTM,
                     'attention_lstm': AttentionLstm,
                     'bayesian_lstm': BayesianLSTM}
        model = model_map[model_type]

        # Model, figure & df name
        lstm_model_name = '_'.join([self.crts_id, model_type, phased_type, 'model.h5'])
        lstm_figure_name = '_'.join([self.crts_id, model_type, phased_type, dc_type, sm_type, 'figure.png'])
        lstm_train_loss_name = '_'.join([model_type, phased_type, dc_type, sm_type, 'train_loss'])
        lstm_cross_loss_name = '_'.join([model_type, phased_type, dc_type, sm_type, 'cross_loss'])

        # Build & Load Model
        lstm = model(hidden_dim, epochs, batch_size, phased_type, n_walkers)
        if train is True:
            print('Fitting %s ' % lstm_model_name)
            lstm_model = lstm.fit_model(self.X_train, self.y_train)
            lstm_model.save(os.path.join(self.model_folder, model_type, lstm_model_name))

        else:
            print('Loading lstm model %s ' % lstm_model_name)
            lstm.model = load_model(os.path.join(self.model_folder, model_type, lstm_model_name),
                                    custom_objects={'PhasedLSTM': PhasedLSTM})

        # Figure & Loss
        fig, train_loss, cross_loss = lstm.prediction(self.t_train, self.mag_train, self.magerr_train, self.X_train,
                                                      self.t_cross, self.mag_cross, self.magerr_cross, self.X_cross,
                                                      self.mag_scaler, self.delta_t_scaler, dc_type, sm_type)
        fig.savefig(os.path.join(self.figure_folder, model_type, lstm_figure_name))

        result_df.loc['id_' + str(self.crts_id), lstm_train_loss_name] = train_loss
        result_df.loc['id_' + str(self.crts_id), lstm_cross_loss_name] = cross_loss
        result_df.to_csv(result_csv)









    def run_carama(self, train=False):
        result_df = pd.read_csv(result_csv, index_col=0, encoding='utf-8')
        # Model & Figure Path
        carma_model_name = self.crts_id + '_carma_model.pkl'
        carma_average_figure_name = self.crts_id + '_carma_average_figure.png'
        carma_sample_figure_name = self.crts_id + '_carma_sample_figure.png'

        carma_train_loss_name = 'carma_train_loss'
        carma_cross_loss_name = 'carma_cross_loss'

        # Model config
        p = self.carma_config["p"]
        q = self.carma_config["q"]
        nwalkers = self.carma_config["nwalkers"]

        # Build & Load Model
        carma = Carma(p, q, nwalkers)

        if train is True:
            print('fitting %s model' % carma_model_name)
            carma_model = carma.fit_model(self.t_train, self.mag_train, self.magerr_train)
            with open(os.path.join(self.model_folder, carma_model_name), 'wb') as handle:
                pickle.dump(carma_model, handle)

        else:
            with open(os.path.join(self.model_folder, carma_model_name), 'rb') as handle:
                carma.model = pickle.load(handle)
                print('carma model %s is loaded' % carma_model_name)

        # Run Sample Simulation
        sample_fig = carma.simulate_sample_process(self.t_train, self.mag_train, self.magerr_train,
                                                   self.t_cross, self.mag_cross, self.magerr_cross)
        sample_fig.savefig(os.path.join(self.carma_figure_folder, carma_sample_figure_name))

        # Run Average Simulation
        average_return = carma.simulate_average_process(self.t_train, self.mag_train, self.magerr_train,
                                                        self.t_cross, self.mag_cross, self.magerr_cross)
        train_loss, cross_loss, average_fig = average_return
        result_df.loc['id_'+str(self.crts_id), carma_train_loss_name] = train_loss
        result_df.loc['id_'+str(self.crts_id), carma_cross_loss_name] = cross_loss
        average_fig.savefig(os.path.join(self.carma_figure_folder, carma_average_figure_name))

        result_df.to_csv(result_csv)

    def run_gp(self, train=True):
        result_df = pd.read_csv(result_csv, index_col=0, encoding='utf-8')
        # Model, figure & df name
        gp_model_name = self.crts_id + '_gp_model.pkl'
        gp_single_figure_name = self.crts_id + '_gp_single_figure.png'
        gp_multi_figure_name = self.crts_id + '_gp_multi_figure.png'

        gp_single_train_loss_name = 'gp_single_train_loss'
        gp_single_cross_loss_name = 'gp_single_cross_loss'
        gp_multi_train_loss_name = 'gp_multi_train_loss'
        gp_multi_cross_loss_name = 'gp_multi_cross_loss'

        # Model config

        # Build & Load Model
        gp = GP()
        if train is True:
            print('fitting %s model' % gp_model_name)
            gp_model = gp.fit_model(self.t_train, self.mag_train, self.magerr_train)
            with open(os.path.join(self.model_folder, gp_model_name), 'wb') as handle:
                pickle.dump(gp_model, handle)

        else:
            with open(os.path.join(self.model_folder, gp_model_name), 'rb') as handle:
                gp.model = pickle.load(handle)
                print('gp model %s is loaded' % gp_model_name)

        # Run Multi Step Simulation
        multi_return = gp.multi_step_prediction(self.t_train, self.mag_train, self.magerr_train,
                                                self.t_cross, self.mag_cross, self.magerr_cross)
        multi_train_loss, multi_cross_loss, multi_fig = multi_return
        result_df.loc['id_'+str(self.crts_id), gp_multi_train_loss_name] = multi_train_loss
        result_df.loc['id_'+str(self.crts_id), gp_multi_cross_loss_name] = multi_cross_loss
        multi_fig.savefig(os.path.join(self.gp_figure_folder, gp_multi_figure_name))

        result_df.to_csv(result_csv)


def mean_std():
    result_df = pd.read_csv(result_csv, index_col=0, encoding='utf-8')
    for i in result_df.columns:
        mean = np.mean(result_df.loc[:, i])
        std = np.std(result_df.loc[:, i])
        result_df.loc['mean', i] = mean
        result_df.loc['std', i] = std
    result_df.to_csv(result_csv)


if __name__ == 'main':
    for crts in crts_list:
        print(crts)
        instance = MainPipeline(crts)
        instance.prepare_individual_data()
        instance.load_individual_data()
