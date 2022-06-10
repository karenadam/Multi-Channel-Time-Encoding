from src import *


class Signal(object):
    def __init__(self):
        return

    def get_total_integral(self, t):
        return np.sum(self.sample(t)) * (t[1] - t[0])


class periodicBandlimitedSignal(Signal):
    def __init__(self, period, n_components, coefficient_values):
        self.period = period
        self._n_components = n_components
        if len(coefficient_values) == n_components:
            extended_coefficients = [x.conjugate() for x in coefficient_values[::-1]]
            extended_coefficients[-1:] = coefficient_values
        else:
            extended_coefficients = coefficient_values
        self.coefficients = extended_coefficients

    def set_coefficients(self, values):
        if len(values) != 2 * self._n_components - 1:
            raise ValueError("You do not have as many coefficients as components")
        self._coefficients = values
        self._frequencies = (
            np.arange(-self._n_components + 1, self._n_components, 1)
            * 2
            * np.pi
            / self.period
        )

    def get_coefficients(self):
        return self._coefficients

    def sample(self, t):
        reshaped_coefficients = np.atleast_2d(self._coefficients).T
        reshaped_frequencies = np.atleast_2d(self._frequencies).T
        reshaped_time = np.atleast_2d(t)
        return np.sum(
            reshaped_coefficients * np.exp(1j * reshaped_frequencies * reshaped_time), 0
        )

    def get_precise_integral(self, t_start, t_end):
        reshaped_coefficients = np.atleast_2d(
            np.delete(self._coefficients, self._n_components - 1)
        ).T
        reshaped_frequencies = np.atleast_2d(
            np.delete(self._frequencies, self._n_components - 1)
        ).T
        integral = np.sum(
            reshaped_coefficients
            / (1j * reshaped_frequencies)
            * np.exp(1j * reshaped_frequencies * t_end)
            - reshaped_coefficients
            / (1j * reshaped_frequencies)
            * np.exp(1j * reshaped_frequencies * t_start),
            0,
        )
        integral += self.coefficients[self._n_components - 1] * (t_end - t_start)
        return np.real(integral)

    coefficients = property(get_coefficients, set_coefficients)


class bandlimitedSignal(Signal):
    def __init__(self, Omega, sinc_locs, sinc_amps=None):
        self._Omega = Omega
        self._sinc_locs = np.array(sinc_locs)
        self._sinc_amps = (
            sinc_amps
            if sinc_amps is not None
            else np.random.random(self._sinc_locs.shape)
        )

    def sample(self, t):
        signal = np.zeros_like(t)
        for i in range(len(self._sinc_locs)):
            signal += self._sinc_amps[i] * Helpers.sinc(
                t - self._sinc_locs[i], self.Omega
            )
        return signal

    def get_precise_integral(self, t_start, t_end):
        t_start = np.atleast_2d(t_start)
        t_end = np.atleast_2d(t_end)
        sinc_locs = np.atleast_2d(self._sinc_locs)
        sinc_amps = np.atleast_2d(self._sinc_amps)
        sinc_amps = np.matlib.repmat(sinc_amps, len(t_start), 1)
        return np.sum(
            sinc_amps
            * (
                Helpers.sinc_integral(t_end.T - sinc_locs, self._Omega)
                - Helpers.sinc_integral(t_start.T - sinc_locs, self._Omega)
            ),
            1,
        )

    def get_sinc_amps(self):
        return self._sinc_amps

    def get_sinc_locs(self):
        return self._sinc_locs

    def get_omega(self):
        return self._Omega

    Omega = property(get_omega)
    sinc_locs = property(get_sinc_locs)
    sinc_amps = property(get_sinc_amps)


class piecewiseConstantSignal(Signal):
    def __init__(self, discontinuities, values):
        if len(discontinuities) != len(values) + 1:
            raise ValueError("The number of discontinuities does not match the number of signal levels")
        self.discontinuities = discontinuities
        self.values = values

    def sample(self,t):
        samples = np.zeros_like(t)
        for l in range(len(self.values)):
            indicator = (np.array(t)>(self.discontinuities[l]))*(np.array(t)<(self.discontinuities[l+1]))
            samples+=self.values[l]*indicator
        return samples

    def low_pass_filter(self, omega):
        return lPFedPCSSignal(self.discontinuities, self.values, omega)


class lPFedPCSSignal(Signal):
    def __init__(self, discontinuities, values, omega):
        self.discontinuities = discontinuities
        self.values = values
        self.omega = omega

    def sample(self, t):
        samples = np.zeros_like(t)
        for value_index in range(len(self.values)):
            samples += self.values[value_index] * (
                Helpers.sinc_integral(t - self.discontinuities[value_index], self.omega)
                - Helpers.sinc_integral(
                    t - self.discontinuities[value_index + 1], self.omega
                )
            )
        return samples

    def project_L_sincs(self, sinc_locs, return_amplitudes=False):
        sinc_amps = self.sample(sinc_locs)
        if return_amplitudes:
            return bandlimitedSignal(self.omega, sinc_locs, sinc_amps), sinc_amps
        else:
            return bandlimitedSignal(self.omega, sinc_locs, sinc_amps)
