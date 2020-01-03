import numpy as np
import matplotlib.pyplot as plt
import carmcmc as cm
from sklearn.metrics import mean_squared_error
import json
np.random.seed(1)


class Carma():
    # Note: This need to be run in python2
    def __init__(self, p, q, nwalkers):
        self.model = None
        self.p = p
        self.q = q
        self.nwalkers = nwalkers

    # def fit_pq(self, t_list_train, mag_list_train, magerr_list_train, p_max=7):
    #     model = cm.CarmaModel(t_list_train, mag_list_train, magerr_list_train)
    #     MLE, pq_list, AICc_list = model.choose_order(p_max, njobs=-1)
    #
    #     return pq_list

    def fit_model(self, t_list_train, mag_list_train, magerr_list_train):
        model = cm.CarmaModel(t_list_train, mag_list_train, magerr_list_train, p=self.p, q=self.q)

        self.model = model.run_mcmc(self.nwalkers)
        self.model.assess_fit()

        return self.model

    def plot_power_spectrum(self, t_list_train, magerr_list_train):
        psd_low, psd_hi, psd_mid, frequencies = self.model.plot_power_spectrum(percentile=95.0, nsamples=5000)
        dt = t_list_train[1:] - t_list_train[:-1]
        noise_level = 2.0 * np.mean(dt) * np.mean(magerr_list_train ** 2)

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

    def simulate_sample_process(self, t_list_train, mag_list_train, magerr_list_train,
                                t_list_cross, mag_list_cross, magerr_list_cross, n_paths=3):

        sample_fig = plt.figure(figsize=(12, 8))
        plt.errorbar(t_list_train, mag_list_train, yerr=magerr_list_train, fmt='k.')
        plt.errorbar(t_list_cross, mag_list_cross, yerr=magerr_list_cross, fmt='k.')

        # Interpolation
        num = (t_list_train.max() - t_list_train.min())/0.2
        t_inter = np.linspace(t_list_train.min(), t_list_train.max(), num=num)
        for i in range(n_paths):
            y_inter = self.model.simulate(t_inter, bestfit='random')
            plt.plot(t_inter, y_inter, color='green', alpha=(1-i/float(n_paths)))

        # Prediction
        num = (t_list_cross.max() - t_list_cross.min())/0.2
        t_fore = np.linspace(t_list_cross.min(), t_list_cross.max(), num=num)
        for i in range(n_paths):
            y_fore = self.model.simulate(t_fore, bestfit='random')
            plt.plot(t_fore, y_fore, color='blue', alpha=(1-i/float(n_paths)))

        # Decoration
        plt.xlim(t_list_train.min(), t_list_cross.max())
        plt.xlabel('Time[days]')
        plt.ylabel('Magnitude')
        plt.title('Simulated Sample Paths')

        return sample_fig

    @staticmethod
    def match_list(model, t_list):
        y_pred_num = int((t_list.max() - t_list.min()) / 0.2)
        t_pred = np.linspace(t_list.min(), t_list.max(), num=y_pred_num)
        y_pred, y_pred_var = model.predict(t_pred)

        y_pred_match = []
        y_pred_var_match = []
        for t in t_list:
            t_id = min(range(len(t_pred)), key=lambda i: abs(t_pred[i] - t))
            y_pred_match.append(y_pred[t_id])
            y_pred_var_match.append(y_pred_var[t_id])

        return t_pred, y_pred, y_pred_var, y_pred_match, y_pred_var_match

    def simulate_average_process(self, t_list_train, mag_list_train, magerr_list_train,
                                 t_list_cross, mag_list_cross, magerr_list_cross):

        average_fig = plt.figure(figsize=(12, 8))
        plt.errorbar(t_list_train, mag_list_train, yerr=magerr_list_train, fmt='k.')
        plt.errorbar(t_list_cross, mag_list_cross, yerr=magerr_list_cross, fmt='k.')

        # Interpolation
        inter_train_match_return = self.match_list(self.model, t_list_train)
        t_inter_train, y_inter_train, y_inter_train_var, y_inter_train_match, y_inter_train_var_match = inter_train_match_return
        train_loss = mean_squared_error(y_inter_train_match, mag_list_train)
        plt.plot(t_inter_train, y_inter_train, color='green', ls='-')
        plt.fill_between(t_inter_train, y1=y_inter_train + np.sqrt(y_inter_train_var),
                         y2=y_inter_train - np.sqrt(y_inter_train_var), color='limegreen', alpha=0.5)

        # Prediction
        pred_cross_match_return = self.match_list(self.model, t_list_cross)
        t_pred_cross, y_pred_cross, y_pred_cross_var, y_pred_cross_match, y_pred_cross_var_match = pred_cross_match_return
        cross_loss = mean_squared_error(y_pred_cross_match, mag_list_cross)
        plt.plot(t_pred_cross, y_pred_cross, color='blue', ls='-')
        plt.fill_between(t_pred_cross, y1=y_pred_cross + np.sqrt(y_pred_cross_var),
                         y2=y_pred_cross - np.sqrt(y_pred_cross_var), color='DodgerBlue', alpha=0.5)

        # Decoration
        plt.xlim(t_list_train.min(), t_list_cross.max())
        plt.xlabel('Time[days]')
        plt.ylabel('Magnitude')
        plt.title('Simulated Average Paths')

        return train_loss, cross_loss, average_fig

