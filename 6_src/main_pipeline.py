import numpy as np
import pandas as pd
import os
import json
import pickle
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt
from utils.data_processor import BasicDataProcessor, LSTMDataProcessor
from global_setting import DATA_FOLDER, lightcurve_list
from global_setting import raw_data_folder, basic_data_folder, carma_data_folder, vanilla_lstm_data_folder
from global_setting import carma_model_folder, vanilla_lstm_model_folder, attention_lstm_model_folder
from global_setting import carma_figure_folder, vanilla_lstm_figure_folder, attention_lstm_figure_folder
# from model.a_carma import Carma
from model.b_gp import GP
from model.c_vanilla_lstm import VanillaLSTM
from model.d_attention_lstm import AttentionLstm
from utils.phased_lstm import PhasedLSTM

result = pd.DataFrame(index=lightcurve_list,
                      columns=['vanilla_standard_single_train_loss', 'vanilla_standard_single_train_loss'
                               'vanilla_standard_single_train_loss', 'vanilla_standard_multi_train_loss',
                               'vanilla_phased_single_train_loss', 'vanilla_phased_multi_train_loss',
                               'attention_standard_single_train_loss', 'attention_standard_multi_train_loss',
                               'attention_phased_single_train_loss', 'attention_phased_multi_train_loss',

                               'vanilla_standard_single_cross_loss', 'vanilla_standard_multi_cross_loss',
                               'vanilla_phased_single_cross_loss', 'vanilla_phased_multi_cross_loss',
                               'attention_standard_single_cross_loss', 'attention_standard_multi_cross_loss',
                               'attention_phased_single_cross_loss', 'attention_phased_multi_cross_loss'])

class MainPipeline:
    def __init__(self, crts_id):
        # Configuration
        self.crts_id = crts_id
        self.laod_data_config()
        self.load_model_config()
        self.load_crts_list()
        # self.prepare_all_data()

        # Basic Data Name
        self.basic_data_name = self.crts_id + '.pkl'

        # Data Folder
        self.raw_data_folder = raw_data_folder
        self.basic_data_folder = basic_data_folder
        self.carma_data_folder = carma_data_folder
        self.vanilla_lstm_data_folder = vanilla_lstm_data_folder

        # LSTM Data Name
        self.vanilla_lstm_window_len = self.vanilla_lstm_model_config["window_len"]
        self.rescaled_mag_name = self.crts_id + '_rescaled_mag' + '.pkl'
        self.mag_scaler_name = self.crts_id + '_mag_scaler' + '.pkl'
        self.rescaled_delta_t_name = self.crts_id + '_rescaled_delta_t' + '.pkl'
        self.delta_t_scaler_name = self.crts_id + '_delta_t_scaler' + '.pkl'
        self.standard_X_y_name = self.crts_id + '_X_y' + '_window_len_' + str(self.vanilla_lstm_window_len) + '.plk'

        # Model Folder
        self.carma_model_folder = carma_model_folder
        self.vanilla_lstm_model_folder = vanilla_lstm_model_folder
        self.attention_lstm_model_folder = attention_lstm_model_folder

        # Figure Folder
        self.carma_figure_folder = carma_figure_folder
        self.vanilla_lstm_figure_folder = vanilla_lstm_figure_folder
        self.attention_lstm_figure_folder = attention_lstm_figure_folder

        # Load Data
        # self.load_individual_data()

    # ----------------------------------- Load Configuration -----------------------------------
    def laod_data_config(self):
        self.data_config = json.load(open('./config/data_config.json'))

    def load_model_config(self):
        self.model_config = json.load(open('./config/model_config.json'))
        self.carma_model_config = self.model_config["carma"]
        self.vanilla_lstm_model_config = self.model_config["vanilla_lstm"]
        self.attention_lstm_model_config = self.model_config["attention_lstm"]

    def load_crts_list(self):
        crts_list = []
        for file in os.listdir(os.path.join(DATA_FOLDER, 'raw_data')):
            if file.endswith(".csv"):
                crts_list.append(file.split('.')[0])
        self.crts_list = crts_list

    # ----------------------------------- Prepare Data -----------------------------------
    def prepare_all_data(self):
        for crts_id in self.crts_list:
            # Basic Data
            basic_data_processor = BasicDataProcessor(crts_id)
            basic_data_processor.prepare_basic_data()

            # LSTM Data
            # lstm_data_processor = LSTMDataProcessor(crts_id)
            # lstm_data_processor.prepare_rescale_mag()
            # lstm_data_processor.prepare_rescale_delta_t()
            # lstm_data_processor.prepare_basic_lstm_data()

    def prepare_individual_data(self):
        # Basic Data
        basic_data_processor = BasicDataProcessor(self.crts_id)
        basic_data_processor.prepare_basic_data()

        # LSTM Data
        lstm_data_processor = LSTMDataProcessor(self.crts_id)
        lstm_data_processor.prepare_rescale_mag()
        lstm_data_processor.prepare_rescale_delta_t()
        lstm_data_processor.prepare_vanilla_lstm_data()

    # ----------------------------------- Load Data -----------------------------------
    def load_individual_data(self):
        # Basic Data
        basic_data_path = os.path.join(self.basic_data_folder, self.basic_data_name)
        with open(basic_data_path, 'rb') as handle:
            data_dict = pickle.load(handle)
        self.mag_list_train = data_dict['mag_list_train']
        self.mag_list_cross = data_dict['mag_list_cross']
        self.mag_list_test = data_dict['mag_list_test']
        self.magerr_list_train = data_dict['magerr_list_train']
        self.magerr_list_cross = data_dict['magerr_list_cross']
        self.magerr_list_test = data_dict['magerr_list_test']
        self.t_list_train = data_dict['t_list_train']
        self.t_list_cross = data_dict['t_list_cross']
        self.t_list_test = data_dict['t_list_test']

        # LSTM Data
        self.mag_scaler = joblib.load(os.path.join(self.vanilla_lstm_data_folder, self.mag_scaler_name))
        with open(os.path.join(self.vanilla_lstm_data_folder, self.standard_X_y_name), 'rb') as handle:
            X_y_data_dict = pickle.load(handle)
            self.train_X = X_y_data_dict['train_X']
            self.train_y = X_y_data_dict['train_y']
            self.cross_X = X_y_data_dict['cross_X']
            self.cross_y = X_y_data_dict['cross_y']
            self.test_X = X_y_data_dict['test_X']
            self.test_y = X_y_data_dict['test_y']

    # ----------------------------------- Run Single Model -----------------------------------
    def run_carama(self):
        # Model & Figure Path
        carma_model_name = self.crts_id + '_carma_model.pkl'
        carma_average_figure_name = self.crts_id + '_carma_average_figure.png'
        carma_sample_figure_name = self.crts_id + '_carma_sample_figure.png'

        # Model config
        p = self.carma_model_config["p"]
        q = self.carma_model_config["q"]
        nwalkers = self.carma_model_config["nwalkers"]

        # Build & Load Model
        carma = Carma(p, q, nwalkers)
        if os.path.isdir(os.path.join(self.carma_model_folder, carma_model_name)):
            with open(os.path.join(self.carma_model_folder, carma_model_name), 'rb') as handle:
                carma.model = pickle.load(handle)
                print('carma model %s is loaded' % carma_model_name)

        else:
            print('fitting %s model' % carma_model_name)
            carma_model = carma.fit_model(self.t_list_train, self.mag_list_train, self.magerr_list_train,
                                          p, q, nwalkers)
            with open(os.path.join(self.carma_model_folder, carma_model_name), 'wb') as handle:
                pickle.dump(carma_model, handle)
                print('carma model %s is loaded' % carma_model_name)

        # Run Average Simulation
        average_figure = carma.simulate_average_process(self.t_list_train, self.mag_list_train, self.magerr_list_train,
                                                        self.t_list_cross, self.mag_list_cross, self.magerr_list_cross)
        average_figure.savefig(os.path.join(self.carma_figure_folder, carma_average_figure_name))

        # Run Sample Simulation
        sample_figure = carma.simulate_sample_process(self.t_list_train, self.mag_list_train, self.magerr_list_train,
                                                      self.t_list_cross, self.mag_list_cross, self.magerr_list_cross)
        sample_figure.savefig(os.path.join(self.carma_figure_folder, carma_sample_figure_name))

    def run_vanilla_lstm(self):

        for phased in ['phased', 'standard']:

            if phased == 'phased':
                phase_name = '_phased'
            else:
                phase_name = '_standard'

            # Model Path
            vanilla_lstm_model_name = self.crts_id + '_vanilla_lstm' + phase_name + '_model.h5'
            vanilla_lstm_single_fit_figure_name = self.crts_id + '_vanilla_lstm' + phase_name + '_single_fit_figure.png'
            vanilla_lstm_single_res_figure_name = self.crts_id + '_vanilla_lstm' + phase_name + '_single_res_figure.png'
            vanilla_lstm_multi_fit_figure_name = self.crts_id + '_vanilla_lstm' + phase_name + '_multi_fit_figure.png'
            vanilla_lstm_multi_res_figure_name = self.crts_id + '_vanilla_lstm' + phase_name + '_multi_res_figure.png'

            # Model config
            window_len = self.vanilla_lstm_model_config["window_len"]
            epochs = self.vanilla_lstm_model_config["epochs"]
            batch_size = self.vanilla_lstm_model_config["batch_size"]
            hidden_dim = self.vanilla_lstm_model_config["hidden_dim"]

            # Build & Load Model
            vanilla_lstm = VanillaLSTM(window_len, hidden_dim, epochs, batch_size, phased=phased)

            if os.path.exists(os.path.join(self.vanilla_lstm_model_folder, vanilla_lstm_model_name)):
                vanilla_lstm.model = load_model(os.path.join(self.vanilla_lstm_model_folder, vanilla_lstm_model_name),
                                                custom_objects={'PhasedLSTM': PhasedLSTM})
                print('vanilla lstm model %s is loaded' % vanilla_lstm_model_name)

            else:
                print('fitting %s model' % vanilla_lstm_model_name)
                vanilla_lstm.build_model()
                vanilla_lstm_model = vanilla_lstm.fit_model(self.train_X, self.train_y, self.cross_X, self.cross_y, self.test_X, self.test_y)
                vanilla_lstm_model.save(os.path.join(self.vanilla_lstm_model_folder, vanilla_lstm_model_name))

            # Run Single Step Simulation
            single_figure = vanilla_lstm.single_step_prediction(self.t_list_train, self.mag_list_train, self.magerr_list_train,
                                                                self.t_list_cross, self.mag_list_cross, self.magerr_list_cross,
                                                                self.train_X, self.cross_X, self.test_X, self.mag_scaler)
            single_fit_figure, single_res_figure = single_figure
            single_fit_figure.savefig(os.path.join(self.vanilla_lstm_figure_folder, vanilla_lstm_single_fit_figure_name))
            single_res_figure.savefig(os.path.join(self.vanilla_lstm_figure_folder, vanilla_lstm_single_res_figure_name))

            # Run Multiple Step Simulation
            multi_figure = vanilla_lstm.multi_step_prediction(self.t_list_train, self.mag_list_train, self.magerr_list_train,
                                                              self.t_list_cross, self.mag_list_cross, self.magerr_list_cross,
                                                              self.train_X, self.cross_X, self.test_X, self.mag_scaler)
            multi_fit_figure, multi_res_figure = multi_figure
            multi_fit_figure.savefig(os.path.join(self.vanilla_lstm_figure_folder, vanilla_lstm_multi_fit_figure_name))
            multi_res_figure.savefig(os.path.join(self.vanilla_lstm_figure_folder, vanilla_lstm_multi_res_figure_name))

    def run_attention_lstm(self, train=True):
        for phased in ['phased', 'standard']:

            if phased == 'phased':
                phase_name = '_phased'
            else:
                phase_name = '_standard'

            # Model Path
            attention_lstm_model_name = self.crts_id + '_attention' + phase_name + '_model.h5'
            attention_lstm_single_fit_figure_name = self.crts_id + '_attention' + phase_name + '_single_fit_figure.png'
            attention_lstm_single_res_figure_name = self.crts_id + '_attention' + phase_name + '_single_res_figure.png'
            attention_lstm_multi_fit_figure_name = self.crts_id + '_attention' + phase_name + '_multi_fit_figure.png'
            attention_lstm_multi_res_figure_name = self.crts_id + '_attention' + phase_name + '_multi_res_figure.png'

            # Model config
            window_len = self.attention_lstm_model_config["window_len"]
            epochs = self.attention_lstm_model_config["epochs"]
            batch_size = self.attention_lstm_model_config["batch_size"]
            hidden_dim = self.attention_lstm_model_config["hidden_dim"]

            # Build & Load Model
            attention_lstm = AttentionLstm(window_len, hidden_dim, epochs, batch_size, phased=phased)

            if train is True:
                print('fitting %s model' % attention_lstm_model_name)
                attention_lstm.build_model()
                attention_lstm_model = attention_lstm.fit_model(self.train_X, self.train_y, self.cross_X, self.cross_y, self.test_X, self.test_y)
                attention_lstm_model.save(os.path.join(self.attention_lstm_model_folder, attention_lstm_model_name))

            else:
                attention_lstm.model = load_model(os.path.join(self.attention_lstm_model_folder, attention_lstm_model_name),
                                                  custom_objects={'PhasedLSTM': PhasedLSTM})
                print('attention lstm model %s is loaded' % attention_lstm_model_name)

            # Run Single Step Simulation
            single_return = attention_lstm.single_step_prediction(self.t_list_train, self.mag_list_train, self.magerr_list_train,
                                                                  self.t_list_cross, self.mag_list_cross, self.magerr_list_cross,
                                                                  self.train_X, self.cross_X, self.test_X, self.mag_scaler)
            single_train_loss, single_cross_loss, single_attention_matrix, single_fit_figure, single_res_figure = single_return
            single_fit_figure.savefig(os.path.join(self.attention_lstm_figure_folder, attention_lstm_single_fit_figure_name))
            single_res_figure.savefig(os.path.join(self.attention_lstm_figure_folder, attention_lstm_single_res_figure_name))

            # Run Multiple Step Simulation
            multi_return = attention_lstm.multi_step_prediction(self.t_list_train, self.mag_list_train, self.magerr_list_train,
                                                                self.t_list_cross, self.mag_list_cross, self.magerr_list_cross,
                                                                self.train_X, self.cross_X, self.test_X, self.mag_scaler)
            multi_train_loss, multi_cross_loss, multi_attention_matrix, multi_fit_figure, multi_res_figure = multi_return
            multi_fit_figure.savefig(os.path.join(self.attention_lstm_figure_folder, attention_lstm_multi_fit_figure_name))
            multi_res_figure.savefig(os.path.join(self.attention_lstm_figure_folder, attention_lstm_multi_res_figure_name))

    # ----------------------------------- Run Hybrid Model -----------------------------------

if __name__ == 'main':


    for lightcurve in lightcurve_list:

        instance = MainPipeline(lightcurve)

        # Prepare Data
        # instance.prepare_all_data()
        # instance.prepare_individual_data()

        # Load Data
        instance.load_individual_data()

        # Run Model
        # instance.run_carama()
        instance.run_vanilla_lstm()
        instance.run_attention_lstm()
