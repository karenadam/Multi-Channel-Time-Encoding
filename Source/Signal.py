import numpy as np


class bandlimitedSignal(object):
    def __init__(self, t, delta_t, Omega, seed=0):
        np.random.seed(int(seed))
        T = np.pi / Omega
        samples_per_period = int(T / delta_t)
        n_sincs = int(len(t) / samples_per_period)
        self.Omega = Omega
        self.sinc_locs = []
        self.sinc_amps = []
        for n in range(n_sincs):
            self.sinc_locs.append(T * (n))
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
