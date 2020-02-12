import numpy as np
from scipy.special import sici
from Helpers import *
import copy


class bandlimitedSignal(object):
    def __init__(self, Omega, sinc_locs=[], sinc_amps=[], padding=0):
        self.n_sincs = len(sinc_locs)
        self.Omega = Omega
        self.sinc_locs = sinc_locs
        self.sinc_amps = sinc_amps


    def random(self, t, padding = 0):
        delta_t = t[1]-t[0]
        T = np.pi / self.Omega
        samples_per_period = int(T / delta_t)
        n_sincs = int(len(t) / samples_per_period)        
        self.n_sincs = n_sincs
        self.sinc_locs = []
        self.sinc_amps = []
        for n in range(n_sincs - 2 * padding):
            self.sinc_locs.append(T * (n + padding))
            a = np.random.uniform(0, 1)
            self.sinc_amps.append(a)
        scale = 2 * np.sqrt(np.mean([a ** 2 for a in self.sinc_amps]))
        self.sinc_amps = self.sinc_amps / scale

    def sample(self, t):
        signal = np.zeros_like(t)
        for i in range(len(self.sinc_locs)):
            signal += (
                self.sinc_amps[i]
                * sinc(t - self.sinc_locs[i], self.Omega)
            )
        return signal

    def get_sincs(self):
        return self.sinc_locs, self.sinc_amps

    def get_total_integral(self, t):
        return np.sum(self.sample(t)) * (t[1] - t[0])

    def get_precise_integral(self, t_start, t_end):
        t_start = np.atleast_2d(t_start)
        t_end = np.atleast_2d(t_end)
        sinc_locs = np.atleast_2d(self.sinc_locs)
        sinc_amps = np.atleast_2d(self.sinc_amps)
        sinc_amps = np.matlib.repmat(sinc_amps, len(t_start), 1)
        return np.sum(sinc_amps*(Si(t_end.T - sinc_locs, self.Omega) - Si(t_start.T-sinc_locs, self.Omega)),1)

    def set_sinc_amps(self, sinc_amps):
        self.sinc_amps = sinc_amps

    def get_sinc_amps(self):
        return self.sinc_amps

    def set_sinc_amps(self, sinc_amps):
        assert sinc_amps.shape == self.sinc_amps.shape
        self.sinc_amps = copy.deepcopy(sinc_amps.tolist())

    def get_sinc_locs(self):
        return self.sinc_locs

    def get_omega(self):
        return self.Omega


class bandlimitedSignals(object):
    def __init__(self,Omega, sinc_locs=[], sinc_amps=[], padding=0):
        self.n_signals = len(sinc_amps)
        self.signals = []
        for n in range(self.n_signals):
            self.signals.append(
                bandlimitedSignal(Omega, sinc_locs, sinc_amps[n])
            )
        self.sinc_locs = sinc_locs
        self.sinc_amps = sinc_amps
        self.Omega = Omega

    def get_n_signals(self):
        return self.n_signals

    def add(self, signal):
        assert signal.Omega == self.Omega
        if len(self.signals) == 0:
            self.signals.append(signal)
            self.sinc_locs = signal.get_sinc_locs()
            self.sinc_amps.append(signal.get_sinc_amps().tolist())
        else:
            assert (signal.get_sinc_locs() == self.sinc_locs)
            self.signals.append(signal)
            self.sinc_amps.append(signal.get_sinc_amps().tolist())
        self.n_signals += 1

    def replace(self, signal, signal_index):
        assert signal_index < len(self.signals)
        assert signal.get_omega() == self.Omega

        assert (signal.get_sinc_locs() == self.sinc_locs)
        self.signals[signal_index] = copy.deepcopy(signal)
        self.sinc_amps[signal_index] = signal.get_sinc_amps().tolist()

    def get_signal(self, signal_index):
        return self.signals[signal_index]
    def get_signals(self):
        return self.signals

    def sample(self, t):
        out = np.zeros((self.n_signals, len(t)))
        for n in range(self.n_signals):
            out[n, :] = self.signals[n].sample(t)
        return out

    def get_sinc_locs(self):
        return self.sinc_locs

    def get_sinc_amps(self):
        return self.sinc_amps

    def get_omega(self):
        return self.Omega

    def set_sinc_amps(self, sinc_amps):
        assert len(sinc_amps) == len(self.sinc_amps)
        assert len(sinc_amps[0]) == len(self.sinc_amps[0])
        for n in range(len(self.signals)):
            self.signals[n].set_sinc_amps(sinc_amps[n])
        self.sinc_amps = copy.deepcopy(sinc_amps)

    def get_integrals(self, t_start, t_end):
        values = [item for sublist in self.sinc_amps for item in sublist]
        return self.get_integral_matrix(t_start, t_end).dot(values)
  

    def get_integral_matrix(self, t_start, t_end):
        assert len(t_start) == self.n_signals
        assert len(t_start)==len(t_end)
        t_start_flattened = [item for sublist in t_start for item in sublist]
        total_num_integrals = len(t_start_flattened)
        values = [item for sublist in self.sinc_amps for item in sublist]
        total_num_values = len(values)
        sinc_locs = np.array(self.sinc_locs)

        sinc_integral_matrix = np.zeros((total_num_integrals, total_num_values))
        matrix_column_index = 0
        matrix_row_index = 0
        for signal_index in range(self.n_signals):
            num_integrals = len(t_start[signal_index])
            integ_up_limit = t_end[signal_index]
            integ_low_limit = t_start[signal_index]
            num_values = len(self.sinc_amps[signal_index])
            for integral_index in range(num_integrals):
                multiplier_vector = Si(integ_up_limit[integral_index] - sinc_locs, self.Omega) - Si(integ_low_limit[integral_index] - sinc_locs, self.Omega)
                sinc_integral_matrix[matrix_row_index, matrix_column_index:matrix_column_index+num_values] = multiplier_vector
                matrix_row_index += 1
            matrix_column_index += num_values

        return sinc_integral_matrix

    

    def mix_amplitudes(self, mixing_matrix):
        values = [item for sublist in self.sinc_amps for item in sublist]

        return self.get_flattened_mixing_matrix(mixing_matrix).dot(values)

    def get_flattened_mixing_matrix(self, mixing_matrix):
        mixing_matrix = np.array(mixing_matrix)
        print(mixing_matrix.shape[1])
        print(self.n_signals)
        assert mixing_matrix.shape[1] == self.n_signals
        values = [item for sublist in self.sinc_amps for item in sublist]
        total_num_values = len(values)
        num_sincs = len(self.sinc_locs)

        flattened_mixing_matrix = np.zeros((num_sincs*mixing_matrix.shape[0], num_sincs*mixing_matrix.shape[1]))
        matrix_column_index = 0
        matrix_row_index = 0
        for signal_index in range(mixing_matrix.shape[0]):
            print(signal_index)
            for sinc_index in range(num_sincs):
                flattened_mixing_matrix[matrix_row_index, sinc_index::num_sincs] = mixing_matrix[signal_index,:]
                matrix_row_index +=1
        return flattened_mixing_matrix

class piecewiseConstantSignal(object):
    def __init__(self, discontinuities, values):
        assert (len(discontinuities) == len(values)+1)
        self.discontinuities = discontinuities
        self.values = values

    def sample(self,t):
        value_index = 0
        samples = np.zeros_like(t)
        for n in range(len(t)):
            if (t[n]<self.discontinuities[0] or t[n]>self.discontinuities[-1]):
                samples[n] = 0
            else:
                while (value_index < len(self.values) and self.discontinuities[value_index+1]<t[n]):
                    value_index += 1
                samples[n] = self.values[value_index]
        return samples

    def low_pass_filter(self, omega):
        return lPFedPCSSignal(self.discontinuities, self.values, omega)

    def get_discontinuities(self):
        return self.discontinuities

    def get_values(self):
        return self.values

class piecewiseConstantSignals(object):
    def __init__(self, discontinuities = [[]], values = [[]]):
        self.discontinuities = discontinuities
        self.values = values
        self.n_signals = len(discontinuities)
        self.signals = []
        for n in range(self.n_signals):
            self.signals.append(
                piecewiseConstantSignal(discontinuities[n], values[n])
            )
    
    def add(self, signal):
        self.signals.append(signal)
        self.discontinuities.append(signal.get_discontinuities())
        self.values.append(signal.get_values())
        self.n_signals += 1

    def replace(self, signal, signal_index):
        self.signals[signal_index] = copy.deepcopy(signal)
        self.discontinuities[signal_index] = signal.get_discontinuities().tolist()
        self.values[signal_index] = signal.get_values().tolist()

    def sample(self, sample_locs, omega):
        values = [item for sublist in self.values for item in sublist]
        samples = self.get_sampler_matrix(sample_locs, omega).dot(values)
        return samples


    def get_sampler_matrix(self, sample_locs, omega):
        num_samples = len(sample_locs)
        sample_locs = np.atleast_2d(sample_locs).T
        values = [item for sublist in self.values for item in sublist]
        total_num_values = len(values)
        values = np.atleast_2d(values).T

        PCS_sampler_matrix = np.zeros((self.n_signals*num_samples, total_num_values))

        matrix_column_index = 0
        matrix_row_index = 0
        for signal_index in range(self.n_signals):
            num_discontinuities = len(self.discontinuities[signal_index])
            num_values = len(self.values[signal_index])
            for sample_index in range(num_samples):
                low_limit = self.discontinuities[signal_index][:-1]
                up_limit = self.discontinuities[signal_index][1:]
                multiplier_vector = Si(sample_locs[sample_index] - low_limit, omega) - Si(sample_locs[sample_index] - up_limit, omega)
                PCS_sampler_matrix[matrix_row_index, matrix_column_index:matrix_column_index+num_values] =multiplier_vector
                matrix_row_index += 1

            matrix_column_index += num_values

        return PCS_sampler_matrix

    def get_signal(self, signal_index):
        return self.signals[signal_index]



class lPFedPCSSignal(object):
    def __init__(self, discontinuities, values, omega):
        self.discontinuities = discontinuities
        self.values = values
        self.omega = omega

    def sample(self, t):
        samples = np.zeros_like(t)
        for value_index in range(len(self.values)):
            samples += self.values[value_index]*(Si(t - self.discontinuities[value_index], self.omega) - Si(t - self.discontinuities[value_index+1], self.omega))
        return samples

    def project_L_sincs(self, sinc_locs, return_amplitudes = False):
        sinc_amps = self.sample(sinc_locs)
        if return_amplitudes:
            return bandlimitedSignal(self.omega, sinc_locs, sinc_amps), sinc_amps
        else:
            return bandlimitedSignal(self.omega, sinc_locs, sinc_amps)


