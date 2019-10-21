import numpy as np


class bandlimitedSignal(object):
    def __init__(self, t, delta_t, Omega, sinc_locs=[], sinc_amps=[], padding=0):
        T = np.pi / Omega
        samples_per_period = int(T / delta_t)
        n_sincs = int(len(t) / samples_per_period)
        self.n_sincs = n_sincs
        self.Omega = Omega
        if len(sinc_locs) > 0 and len(sinc_amps) > 0:
            self.sinc_locs = sinc_locs
            self.sinc_amps = sinc_amps
        else:
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
                * np.sinc(self.Omega * (t - self.sinc_locs[i]) / np.pi)
                * self.Omega
                / np.pi
            )
        return signal

    def get_sincs(self):
        return self.sinc_locs, self.sinc_amps

    def get_total_integral(self, t):
        return np.sum(self.sample(t)) * (t[1] - t[0])

    def set_sinc_amps(self, sinc_amps):
        self.sinc_amps = sinc_amps

    def get_sinc_amps(self):
        return self.sinc_amps


class bandlimitedSignals(object):
    def __init__(self, t, delta_t, Omega, sinc_locs=[], sinc_amps=[], padding=0):
        self.n_signals = len(sinc_amps)
        self.signals = []
        for n in range(self.n_signals):
            self.signals.append(
                bandlimitedSignal(t, delta_t, Omega, sinc_locs, sinc_amps[n])
            )
        self.sinc_locs = sinc_locs

    def get_signals(self):
        return self.signals

    def sample(self, t):
        out = np.zeros((self.n_signals, len(t)))
        for n in range(self.n_signals):
            out[n, :] = self.signals[n].sample(t)
        return out
