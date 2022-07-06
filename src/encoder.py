import numpy as np
import scipy.optimize
import bisect
from src import *


class Encoder(object):
    """
    Base class for encoder objects

    ATTRIBUTES
    ----------
    tem_parameters: TEMParams
        holds the parameters of this object
    with_integral_probe: bool
        determines whether or not this encode has access to the output
        of the integrator(s) (True) as well, or just the spike times (False)
    """

    def __init__(self, tem_parameters, with_integral_probe=False):
        """
        PARAMETERS
        ----------
        tem_parameters: TEMParams
            holds the parameters of this object
        with_integral_probe: bool
            determines whether or not this encode has access to the output
            of the integrator(s) (True) as well, or just the spike times (False)
        """

        self.params = tem_parameters
        self.with_integral_probe = with_integral_probe

    def _get_weighted_integral(
        self,
        y_param,
        ch,
        start_time,
        end_time,
    ):
        integral = y_param[ch].get_precise_integral(start_time, end_time)
        weighted_integral = (
            integral + self.params._b[ch] * (end_time - start_time)
        ) / self.params._kappa[ch]
        if start_time == 0:
            weighted_integral += self.params.integrator_init[ch] + self.params.delta[ch]
        return weighted_integral


class DiscreteEncoder(Encoder):
    """
    Class that allows the encoding of a or multiple signals using
    integrate-and-fire time encoding machines in a discrete way, i.e.
    by approximating the integral of the signal by a sum

    ATTRIBUTES
    ----------
    tem_parameters: TEMParams
        holds the parameters of this object
    with_integral_probe: bool
        determines whether or not this encode has access to the output
        of the integrator(s) (True) as well, or just the spike times (False)
    """

    def encode(self, signal, signal_end_time, delta_t):
        """
        encodes the given signal(s) using the different channels of the encoder
        with a spike time output that is generated according to an integrate-
        and-fire time encoding scheme

        PARAMETERS
        ----------
        signal: np.ndarray or Signal.signalCollection
            signal(s) to be encoded
        signal_end_time: float
            time at which encoding should stop
        delta_t: float
            time step for integration

        RETURNS
        -------
        SpikeTimes
            spikeTime object containing the spikes emitted by the
            different channels
        np.ndarray (optional)
            output of integrator throughout the encoding scheme if the
            with_integrator_probe option is turned on
        """

        self.__dict__.update(self.params.__dict__)

        spikes = SpikeTimes(self.n_channels)
        if isinstance(signal, src.signals.Signal) or isinstance(
            signal, src.signals.signalCollection
        ):
            sampled = signal.sample(np.arange(0, signal_end_time, delta_t))
        else:
            sampled = signal
        time = np.arange(0, signal_end_time, delta_t)

        weighted_biased_integral = (
            (
                self.mixing_matrix.dot(np.cumsum(np.atleast_2d(sampled), 1) * delta_t)
                + np.outer(self._b, time)
            ).T
            / np.array(self._kappa)
            + np.array(self._integrator_init)
            + self._delta
        )
        moduloed_integral_quotient = np.floor_divide(
            weighted_biased_integral, 2 * np.array(self._delta)
        ).T

        for ch in range(self.n_channels):
            unique, unique_indices = np.unique(
                moduloed_integral_quotient[ch, :], return_index=True
            )
            start_index = bisect.bisect_right(unique, 0)
            unique = unique[start_index:]
            unique_indices = unique_indices[start_index:]
            if (np.diff(unique_indices) < 0).any() or (
                np.diff(moduloed_integral_quotient) > 1
            ).any():
                print(delta_t)
                raise ValueError("Your delta_t is too large")
            spikes.add(ch, time[unique_indices].tolist())
        return spikes


class ContinuousEncoder(Encoder):
    """
    Class that allows the encoding of a or multiple signals using
    integrate-and-fire time encoding machines in a continuous way, i.e.
    assuming a parametric model for the signal and finding the spike times
    using binary search

    ATTRIBUTES
    ----------
    tem_parameters: TEMParams
        holds the parameters of this object
    with_integral_probe: bool
        determines whether or not this encoder has access to the output
        of the integrator(s) (True) as well, or just the spike times (False)
    """

    def encode(
        self,
        x_param,
        signal_end_time,
        tolerance=1e-8,
    ):
        """
        performs the encoding of the input signals at channel ch of the encoder
        with a spike time output that is generated according to an integrate-
        and-fire time encoding scheme

        PARAMETERS
        ----------
        x_param: Signal.signalCollection
            signal(s) to be encoded
        signal_end_time: float
            time at which encoding should stop
        tolerance: float
            tolerated error on the spike times

        RETURNS
        -------
        SpikeTimes
            spikeTime object holding the timing of the spikes of each of
            the channels
        """

        self.__dict__.update(self.params.__dict__)

        discrete_encoder = DiscreteEncoder(self.params)
        # TODO need to find smarter way of setting delta_t, also depending on size of signal and b
        delta_t = (2 * np.pi / x_param.max_frequency) / 100
        approx_spikes = discrete_encoder.encode(x_param, signal_end_time, delta_t)

        y_param = x_param.get_mixed_signals(self.mixing_matrix)
        spikes = SpikeTimes(self.n_channels)

        def fun(
            integral_end_time,
            signal,
            channel,
            integral_start_time,
        ):
            weighted_integral = self._get_weighted_integral(
                signal,
                channel,
                integral_start_time,
                integral_end_time,
            )
            return (2 * self._delta[channel] - weighted_integral) ** 2

        for ch in range(self.n_channels):
            last_spike = 0
            spikes_of_ch = approx_spikes[ch]

            for i_s, s in enumerate(spikes_of_ch):
                next_spike = scipy.optimize.minimize(
                    fun,
                    s,
                    (y_param, ch, last_spike),
                    bounds=[
                        (
                            spikes_of_ch[i_s - 1] if i_s > 0 else 0,
                            spikes_of_ch[i_s + 1]
                            if i_s < len(spikes_of_ch) - 1
                            else signal_end_time,
                        )
                    ],
                    tol=tolerance,
                ).x[0]
                spikes.add(ch, next_spike)
                last_spike = next_spike
        return spikes
