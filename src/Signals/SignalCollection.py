from src import *


class SignalCollection(object):
    def __init__(self):
        self._n_signals = 0
        self._signals = []

    def add(self, signal):
        self._signals.append(signal)
        self._n_signals += 1

    def __getitem__(self, signal_index):
        return self._signals[signal_index]

    def get_n_signals(self):
        return self._n_signals

    def sample(self, t):
        out = np.zeros((self._n_signals, len(t)))
        for n in range(self._n_signals):
            out[n, :] = self._signals[n].sample(t)
        return out

    n_signals = property(get_n_signals)


class periodicBandlimitedSignals(SignalCollection):
    def __init__(self, period, n_components=0, coefficient_values=None):
        self.period = period
        self._n_components = n_components
        self._n_signals = (
            len(coefficient_values) if coefficient_values is not None else 0
        )
        self._signals = [
            Signal.periodicBandlimitedSignal(
                period, n_components, coefficient_values[n]
            )
            for n in range(self._n_signals)
        ]
        self.coefficient_values = (
            coefficient_values if coefficient_values is not None else []
        )

    def add(self, signal):
        if signal.period != self.period:
            raise ValueError("The period of the signal you are adding does not match the period of the collection")
        if len(self._signals) == 0:
            self._n_components = signal._n_components
        elif self._n_components != signal._n_components:
            raise ValueError("The number of components of the signal you are adding does not match that of the collection")
        self.coefficient_values.append(signal.coefficients)
        super().add(signal)

    def get_mixed_signals(self, mixing_matrix):
        new_coefficients = mixing_matrix.dot(self.coefficient_values)
        new_signals = periodicBandlimitedSignals(
            self.period, self._n_components, new_coefficients
        )
        return new_signals

class bandlimitedSignals(SignalCollection):
    def __init__(self, Omega, sinc_locs=None, sinc_amps=None):
        self._n_signals = len(sinc_amps) if sinc_amps is not None else 0
        self._signals = [
            Signal.bandlimitedSignal(Omega, sinc_locs, sinc_amps[n])
            for n in range(self._n_signals)
        ]
        self._sinc_locs = np.array(sinc_locs) if sinc_locs is not None else []
        self._sinc_amps = sinc_amps if sinc_amps is not None else []
        self._omega = Omega

    def add(self, signal):
        if signal.Omega != self._omega:
            raise ValueError("The bandwidth of the signal you are adding does not match the bandwidth of the collection")
        if len(self._signals) == 0:
            self._sinc_locs = copy.deepcopy(signal.get_sinc_locs())
        elif not np.allclose(signal.get_sinc_locs(), self._sinc_locs):
                raise ValueError(
                    "Locations of sincs of added signal do not match that of collection"
                )
        self._sinc_amps.append(signal.get_sinc_amps().tolist())
        super().add(signal)

    def mix_amplitudes(self, mixing_matrix):
        return np.array(mixing_matrix).dot(np.array(self._sinc_amps)).flatten()

    def get_mixed_signals(self, mixing_matrix):
        new_amps = mixing_matrix.dot(self._sinc_amps)
        new_signals = bandlimitedSignals(self._omega, self._sinc_locs, new_amps)
        return new_signals

    def get_sinc_locs(self):
        return self._sinc_locs

    def get_sinc_amps(self):
        return self._sinc_amps

    def get_omega(self):
        return self._omega

    sinc_locs = property(get_sinc_locs)
    sinc_amps = property(get_sinc_amps)
    omega = property(get_omega)


class piecewiseConstantSignals(SignalCollection):
    def __init__(self, discontinuities=[[]], values=[[]]):
        self.discontinuities = discontinuities
        self.values = values
        self._n_signals = len(discontinuities)
        self._signals = [
            Signal.piecewiseConstantSignal(discontinuities[n], values[n])
            for n in range(self._n_signals)
        ]

    def add(self, signal):
        self.discontinuities.append(signal.get_discontinuities())
        self.values.append(signal.get_values())
        super().add(signal)

    def sample(self, sample_locs, omega):
        values = [item for sublist in self.values for item in sublist]
        samples = self.get_sampler_matrix(sample_locs, omega).dot(values)
        return samples

    def get_sampler_matrix(self, sample_locs, omega):
        def multiplier_vector(sample_loc, signal_index):
            low_limit = np.array(self.discontinuities[signal_index][:-1])
            up_limit = np.array(self.discontinuities[signal_index][1:])
            return np.atleast_2d(
                Helpers.sinc_integral(sample_loc - low_limit, omega)
                - Helpers.sinc_integral(sample_loc - up_limit, omega)
            )

        PCS_sampler_matrix = scipy.linalg.block_diag(
            *[
                np.concatenate(
                    [
                        multiplier_vector(sample_loc, signal_index)
                        for sample_loc in sample_locs
                    ]
                )
                for signal_index in range(self._n_signals)
            ]
        )
        return PCS_sampler_matrix

