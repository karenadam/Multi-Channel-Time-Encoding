import numpy as np
import numpy.matlib
import src
import src.helpers.kernels
from src import *


class Signal(object):
    """Parametric representation of a signal"""

    def __init__(self):
        return

    def get_total_integral(self, t):
        return np.sum(self.sample(t)) * (t[1] - t[0])


class periodicBandlimitedSignal(Signal):
    """periodic bandlimited signal which is described using complex exponentials and
    their corresponding parameters

    ATTRIBUTES
    ----------
    period: float
        period of the signal
    _n_components: int
        number of components in the periodic signal, where the (complex conjugate)
        coefficient values are stored for components -n_components+1 to n_components-1
    _coefficients: list
        coefficients of the complex exponentials that satisfy conjugate symmetry
    _frequencies: frequencies of the complex exponentials for every component k where
        the frequency satisfies f_k = 2\pi k/ period
    """

    def __init__(self, period, n_components, coefficient_values):
        """
        PARAMETERS
        ----------
        period: float
            period of the signal
        n_components: int
            number of components in the periodic signal, where the (complex conjugate)
            coefficient values are stored for components -n_components+1 to n_components-1
        coefficient_values: list
            coefficients of the complex exponentials that satisfy conjugate symmetry
        """
        self.period = period
        self._n_components = n_components
        self._frequencies = (
            np.arange(-self._n_components + 1, self._n_components, 1)
            * 2
            * np.pi
            / self.period
        )
        self._max_frequency = max(self._frequencies)
        self.coefficients = coefficient_values

    def set_coefficients(self, values):
        """
        sets the coefficients and frequencies of the complex exponentials

        PARAMETERS
        ----------
        values: list
            coefficients of the complex exponentials (providing coefficients either from 0 or from
            -n_coefficients+1 to n_coefficients-1
        """

        if len(values) == self._n_components:
            extended_coefficients = [x.conjugate() for x in values[::-1]]
            extended_coefficients[-1:] = values
        elif len(values) != 2 * self._n_components - 1:
            raise ValueError("You do not have as many coefficients as components")
        else:
            extended_coefficients = values
        self._coefficients = extended_coefficients

    def get_coefficients(self):
        """
        RETURNS
        -------
        list
            coefficients of the signal
        """
        return self._coefficients

    def sample(self, t):
        """
        PARAMETERS
        ----------
        t: float or array_like
            time(s) at which the signal should be sampled

        RETURNS
        -------
        np.ndarray
            samples of signal at time(s) t
        """

        reshaped_coefficients = np.atleast_2d(self._coefficients).T
        reshaped_frequencies = np.atleast_2d(self._frequencies).T
        reshaped_time = np.atleast_2d(t)
        return np.sum(
            reshaped_coefficients * np.exp(1j * reshaped_frequencies * reshaped_time), 0
        )

    def get_precise_integral(self, t_start, t_end):
        """
        PARAMETERS
        ----------
        t_start: float or array_like
            lower bound(s) of the integral computation
        t_end: float or array_like
            upper bound(s) of the integral computation

        RETURNS
        -------
        np.ndarray
            integral of the signal between time(s) t_start and t_end
        """

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
        # integral = np.sum(
        #     reshaped_coefficients
        #     / (1j * reshaped_frequencies)
        #     * np.exp(1j * reshaped_frequencies * t_end)
        #     - reshaped_coefficients
        #     / (1j * reshaped_frequencies)
        #     * np.exp(1j * reshaped_frequencies * t_start),
        #     0,
        # )
        integral += self.coefficients[self._n_components - 1] * (t_end - t_start)
        return np.real(integral)

    def get_max_frequency(self):
        return self._max_frequency

    coefficients = property(get_coefficients, set_coefficients)
    max_frequency = property(get_max_frequency)


class bandlimitedSignal(Signal):
    """
    bandlimited signal which is described using a sum of sincs with defined amplitudes

    ATTRIBUTES
    ----------
    _Omega: float
        bandwith of the signal, i.e. frequency used for the sincs
    _sinc_locs: np.ndarray
        locations of the sincs that form the signal
    _sinc_amps: np.ndarray
        amplitudes of the sincs that form the signal
    """

    def __init__(self, Omega, sinc_locs, sinc_amps=None):
        """
        PARAMETERS
        ----------
        Omega: float
            bandwith of the signal, i.e. frequency used for the sincs
        sinc_locs: array_like
            locations of the sincs that form the signal
        sinc_amps: array_like
            amplitudes of the sincs that form the signal
        """

        self._Omega = Omega
        self._sinc_locs = np.array(sinc_locs)
        self._sinc_amps = (
            sinc_amps
            if sinc_amps is not None
            else np.random.random(self._sinc_locs.shape)
        )

    def sample(self, t):
        """
        PARAMETERS
        ----------
        t: float or array_like
            time(s) at which the signal should be sampled

        RETURNS
        -------
        np.ndarray
            samples of signal at time(s) t
        """

        signal = np.zeros_like(t)
        for i in range(len(self._sinc_locs)):
            signal += self._sinc_amps[i] * src.helpers.kernels.sinc(
                t - self._sinc_locs[i], self.Omega
            )
        return signal

    def get_precise_integral(self, t_start, t_end):
        """
        PARAMETERS
        ----------
        t_start: float or array_like
            lower bound(s) of the integral computation
        t_end: float or array_like
            upper bound(s) of the integral computation

        RETURNS
        -------
        np.ndarray
            integral of the signal between time(s) t_start and t_end
        """

        t_start = np.atleast_2d(t_start)
        t_end = np.atleast_2d(t_end)
        sinc_locs = np.atleast_2d(self._sinc_locs)
        sinc_amps = np.atleast_2d(self._sinc_amps)
        sinc_amps = np.matlib.repmat(sinc_amps, len(t_start), 1)
        return np.sum(
            sinc_amps
            * (
                src.helpers.kernels.sinc_integral(t_end.T - sinc_locs, self._Omega)
                - src.helpers.kernels.sinc_integral(t_start.T - sinc_locs, self._Omega)
            ),
            1,
        )

    def get_sinc_amps(self):
        return self._sinc_amps

    def get_sinc_locs(self):
        return self._sinc_locs

    def get_omega(self):
        return self._Omega

    def get_max_frequency(self):
        return self._max_frequency

    Omega = property(get_omega)
    sinc_locs = property(get_sinc_locs)
    sinc_amps = property(get_sinc_amps)
    max_frequency = property(get_omega)


class piecewiseConstantSignal(Signal):
    """
    (1-dimensional) Piecewise Constant Signal defined by a set of discontinuity points
    and a set of values taken between pairs of successive discontinuities

    ATTRIBUTES
    ----------
    _discontinuities: array_like
        points at which the signal changes value
    _values: array_like
        values the signal takes between pairs of successive discontinuities
    """

    def __init__(self, discontinuities, values):
        """
        PARAMETERS
        ----------
        discontinuities: array_like
            points at which the signal changes value
        values: array_like
            values the signal takes between pairs of successive discontinuities
        """
        if len(discontinuities) != len(values) + 1:
            raise ValueError(
                "The number of discontinuities does not match the number of signal levels"
            )
        self._discontinuities = discontinuities
        self._values = values

    def sample(self, t):
        """
        PARAMETERS
        ----------
        t: float or array_like
            time(s) at which the signal should be sampled

        RETURNS
        -------
        np.ndarray
            samples of signal at time(s) t
        """

        samples = np.zeros_like(t)
        for l in range(len(self.values)):
            indicator = (np.array(t) > (self.discontinuities[l])) * (
                np.array(t) < (self.discontinuities[l + 1])
            )
            samples += self.values[l] * indicator
        return samples

    def low_pass_filter(self, omega):
        """
        PARAMETERS
        ----------
        omega: float
            desired bandwidth of low pass filtered version of this signal

        RETURNS
        -------
        lPFedPCSSignal
            low pass filtered version of this signal with bandwidth omega
        """
        return lPFedPCSSignal(self.discontinuities, self.values, omega)

    def get_discontinuities(self):
        return self._discontinuities

    def get_values(self):
        return self._values

    discontinuities = property(get_discontinuities)
    values = property(get_values)


class lPFedPCSSignal(Signal):
    """
    (1-dimensional) low pass filtered piecewise constant signal defined by a set of
    discontinuity points and a set of values taken between pairs of successive discontinuities,
    and filtered using a sinc of bandwidth omega

    ATTRIBUTES
    ----------
    _discontinuities: array_like
        points at which the signal changes value
    _values: array_like
        values the signal takes between pairs of successive discontinuities
    omega: float
        bandwidth of sinc used to low pass filter the signal
    """

    def __init__(self, discontinuities, values, omega):
        self._discontinuities = discontinuities
        self._values = values
        self.omega = omega

    def sample(self, t):
        """
        PARAMETERS
        ----------
        t: float or array_like
            time(s) at which the signal should be sampled

        RETURNS
        -------
        np.ndarray
            samples of signal at time(s) t
        """

        samples = np.zeros_like(t)
        for value_index in range(len(self.values)):
            samples += self.values[value_index] * (
                src.helpers.kernels.sinc_integral(
                    t - self.discontinuities[value_index], self.omega
                )
                - src.helpers.kernels.sinc_integral(
                    t - self.discontinuities[value_index + 1], self.omega
                )
            )
        return samples

    def project_L_sincs(self, sinc_locs):
        """
        projects the signal onto the convex set of sums of sincs at given locations

        PARAMETERS
        ----------
        sinc_locs: array_like
            location of sincs that make up the resulting signal

        RETURNS
        -------
        bandlimitedSignal
            projection of this signal onto the convex set of sums of sincs at sinc_locs
        """
        sinc_amps = self.sample(sinc_locs)
        return bandlimitedSignal(self.omega, sinc_locs, sinc_amps)

    def get_discontinuities(self):
        return self._discontinuities

    def get_values(self):
        return self._values

    discontinuities = property(get_discontinuities)
    values = property(get_values)
