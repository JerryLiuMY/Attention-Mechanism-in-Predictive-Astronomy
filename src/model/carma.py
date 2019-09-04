import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import DataLoader
import carmcmc as cm
import json
np.random.seed(1)


class Carma():
    # Note: This need to be run in python2
    def __init__(self, data_dict):
        self.mag_list_train = data_dict['mag_list_train']
        self.mag_list_cross = data_dict['mag_list_cross']
        self.mag_list_test = data_dict['mag_list_test']
        self.magerr_list_train = data_dict['magerr_list_train']
        self.magerr_list_cross = data_dict['magerr_list_cross']
        self.magerr_list_test = data_dict['magerr_list_test']
        self.t_list_train = data_dict['t_list_train']
        self.t_list_cross = data_dict['t_list_cross']
        self.t_list_test = data_dict['t_list_test']
        self.crts_id = data_dict['crts_id']
        self.data_config = json.load(open('./config/data_config.json'))
        self.model_config = json.load(open('./config/model_config.json'))
        self.sample = None

    def fit_pq(self, p_max=7):
        model = cm.CarmaModel(self.t_list_train, self.mag_list_train, self.magerr_list_train)
        MLE, pq_list, AICc_list = model.choose_order(p_max, njobs=-1)

        return pq_list

    def fit_model(self):
        model = cm.CarmaModel(self.t_list_train, self.mag_list_train, self.magerr_list_train,
                              p=self.model_config["carma"]["p"], q=self.model_config["carma"]["q"])

        self.sample = model.run_mcmc(self.model_config["carma"]["nwalkers"])
        self.sample.assess_fit()

        return self.sample

    def power_spectrum(self):
        psd_low, psd_hi, psd_mid, frequencies = self.sample.plot_power_spectrum(percentile=95.0, nsamples=5000)
        dt = self.t_list_train[1:] - self.t_list_train[:-1]
        noise_level = 2.0 * np.mean(dt) * np.mean(self.magerr_list_train ** 2)

        plt.loglog(frequencies, psd_mid)
        plt.fill_between(frequencies, psd_hi, y2=psd_low, alpha=0.5)
        plt.loglog(frequencies, np.ones(frequencies.size) * noise_level, color='grey', lw=2)

        # Decoration
        plt.ylim(noise_level / 10.0, plt.ylim()[1])
        plt.xlim(frequencies.min(), frequencies[psd_hi > noise_level].max() * 10.0)
        plt.xlabel('Frequency [1 / day]')
        plt.ylabel('Power Spectrum')
        plt.annotate('Measurement Noise Level', (1.25 * plt.xlim()[0], noise_level / 1.5))
        plt.show()

    def sample_interpolation(self, n_paths=3):
        # Training plot
        plt.errorbar(self.t_list_train, self.mag_list_train, yerr=self.magerr_list_train, fmt='k.')

        # Interpolation plot
        num = (self.t_list_train.max() - self.t_list_train.min())/0.2
        t_inter = np.linspace(self.t_list_train.min(), self.t_list_train.max(), num=num)
        for i in range(n_paths):
            y_inter = self.sample.simulate(t_inter, bestfit='random')
            plt.plot(t_inter, y_inter)

        # Decoration
        plt.xlim(self.t_list_train.min(), self.t_list_train.max())
        plt.xlabel('Time [days]')
        plt.ylabel('Magnitude')
        plt.title('Simulated Interpolation Paths')

    def average_forecasting(self):
        # Training plot
        plt.errorbar(self.t_list_train, self.mag_list_train, yerr=self.magerr_list_train, fmt='k.')

        # Cross validation plot
        plt.errorbar(self.t_list_cross, self.mag_list_cross, yerr=self.magerr_list_cross, fmt='k.')

        # Forecasting plot
        num = (self.t_list_cross.max() - self.t_list_cross.min())/0.2
        t_fore = np.linspace(self.t_list_cross.min(), self.t_list_cross.max(), num=num)
        y_fore, y_var = self.sample.predict(t_fore)
        plt.plot(t_fore, y_fore, 'b-')
        plt.fill_between(t_fore, y1=y_fore + np.sqrt(y_var), y2=y_fore - np.sqrt(y_var), color='DodgerBlue', alpha=0.5)

        # Decoration
        plt.xlim(self.t_list_train.min(), self.t_list_cross.max())
        plt.xlabel('Time[days]')
        plt.ylabel('Magnitude')
        plt.title('Expected Forecasting Value')

    def sample_forecasting(self, n_paths=3):
        # Training plot
        plt.errorbar(self.t_list_train, self.mag_list_train, yerr=self.magerr_list_train, fmt='k.')

        # Cross validation plot
        plt.errorbar(self.t_list_cross, self.mag_list_cross, yerr=self.magerr_list_cross, fmt='k.')

        # Forecasting plot
        num = (self.t_list_cross.max() - self.t_list_cross.min())/0.2
        t_fore = np.linspace(self.t_list_cross.min(), self.t_list_cross.max(), num=num)
        for i in range(n_paths):
            y_fore = self.sample.simulate(t_fore, bestfit='random')
            plt.plot(t_fore, y_fore)

        # Decoration
        plt.xlim(self.t_list_train.min(), self.t_list_cross.max())
        plt.xlabel('Time[days]')
        plt.ylabel('Magnitude')
        plt.title('Simulated Forecasting Paths')
