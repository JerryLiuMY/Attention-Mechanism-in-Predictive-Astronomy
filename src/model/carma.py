import numpy as np
import matplotlib.pyplot as plt
import carmcmc as cm
from utils.tools import average_plot, sample_plot, match_list
from sklearn.metrics import mean_squared_error
np.random.seed(1)


class Carma:
    # Note: This need to be run in python2
    def __init__(self, p, q, nwalkers):
        self.p = p
        self.q = q
        self.nwalkers = nwalkers
        self.model = None

    def fit_model(self, t_train, mag_train, magerr_train):
        model = cm.CarmaModel(t_train, mag_train, magerr_train, p=self.p, q=self.q)
        self.model = model.run_mcmc(self.nwalkers)
        self.model.assess_fit()

    def continuous_prediction(self, t, mag, magerr, n):
        y_pred_num = int((t.max() - t.min()) / 0.2)
        t_pred = np.linspace(t.min(), t.max(), num=y_pred_num)
        y_pred, y_pred_var = self.model.predict(t_pred)

        y_pred_n = []
        for i in range(n):
            y_pred, y_pred_var = self.model.predict(t_pred)
            y_pred_n.append(y_pred)

        y_pred_match, y_pred_var_match = match_list(t, t_pred, y_pred, y_pred_var)
        loss = mean_squared_error(mag, y_pred_match)

        return t_pred, y_pred, y_pred_n, y_pred_var, loss

    def plot(self, t_train, mag_train, magerr_train, t_pred_train, y_pred_train, y_pred_train_n, y_pred_var_train,
             t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, y_pred_cross_n, y_pred_var_cross):

        average_fig = average_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train, y_pred_var_train,
                                   t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross, y_pred_var_cross)

        sample_fig = sample_plot(t_train, mag_train, magerr_train, t_pred_train, y_pred_train_n,
                                 t_cross, mag_cross, magerr_cross, t_pred_cross, y_pred_cross_n)

        return average_fig, sample_fig

    def plot_power_spectrum(self, t_train, magerr_train):
        psd_low, psd_hi, psd_mid, frequencies = self.model.plot_power_spectrum(percentile=95.0, nsamples=5000)
        dt = t_train[1:] - t_train[:-1]
        noise_level = 2.0 * np.mean(dt) * np.mean(magerr_train ** 2)

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

    @staticmethod
    def fit_pq(t_train, mag_train, magerr_train, p_max=7):
        model = cm.CarmaModel(t_train, mag_train, magerr_train)
        MLE, pq_list, AICc_list = model.choose_order(p_max, njobs=-1)

        return pq_list
