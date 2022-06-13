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
        signal: np.ndarray or Signal.SignalCollection
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
        if isinstance(signal, Signal.Signal) or isinstance(
            signal, SignalCollection.SignalCollection
        ):
            sampled = signal.sample(np.arange(0, signal_end_time, delta_t))
        else:
            sampled = signal
        input_signal = self.mixing_matrix.dot(np.atleast_2d(sampled))
        if self.with_integral_probe:
            integrator_output = np.zeros((self.n_channels, max(signal.shape)))
        for ch in range(self.n_channels):
            spike_times, run_sum = self.encode_channel(input_signal, ch, delta_t)
            spikes.add(ch, spike_times)
            if self.with_integral_probe:
                integrator_output[ch, :] = run_sum
        if self.with_integral_probe:
            return spikes, integrator_output
        return spikes

    def encode_channel(self, input_signal, ch, delta_t):
        """
        performs the encoding of the input signals at channel ch of the encoder
        with a spike time output that is generated according to an integrate-
        and-fire time encoding scheme

        PARAMETERS
        ----------
        input_signal: np.ndarray
            sampled version of signal(s) to be encoded
        ch: int
            index of the channel of interest
        delta_t: float
            time step for integration

        RETURNS
        -------
        list
            list of floats representing the spike times of the channel
        np.ndarray
            output of integrator throughout the encoding scheme
        """
        spike_locations = []
        spike_times = []
        input_to_ch = input_signal[ch, :]
        run_sum = np.cumsum(delta_t * (input_to_ch + self._b[ch])) / self._kappa[ch]
        thresh = self._delta[ch] - self._integrator_init[ch]
        nextpos = bisect.bisect_left(run_sum, thresh)
        while nextpos != len(input_to_ch):
            spike_locations.append(nextpos)
            spike_times.append(float(nextpos * delta_t))
            thresh = thresh + 2 * self._delta[ch]
            nextpos = bisect.bisect_left(run_sum, thresh)
        if self.with_integral_probe:
            run_sum += self._integrator_init[ch]
            for spike_loc in spike_locations:
                run_sum[spike_loc:] -= 2 * self._delta[ch]
        return spike_times, run_sum


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
        determines whether or not this encode has access to the output
        of the integrator(s) (True) as well, or just the spike times (False)
    """

    def encode_single_channel_precise(
        self,
        signal,
        signal_end_time,
        ch,
        tolerance=1e-6,
        with_start_time=False,
    ):
        """
        performs the encoding of the input signals at channel ch of the encoder
        with a spike time output that is generated according to an integrate-
        and-fire time encoding scheme

        PARAMETERS
        ----------
        signal: Signal.SignalCollection
            signal(s) to be encoded
        signal_end_time: float
            time at which encoding should stop
        ch: int
            index of the channel of interest
        tolerance: float
            tolerated error on the spike timess
        with_start_time: bool
            determines whether or not the starting time of the time encoding
            is included as the first spike time

        RETURNS
        -------
        list
            list of floats representing the spike times of the channel
        """
        z = [0]
        prvs_integral = 0
        current_int_end = signal_end_time
        upp_int_bound = signal_end_time
        low_int_bound = 0
        if signal_end_time == 0:
            return []
        while z[-1] < signal_end_time:
            si = (
                signal.get_precise_integral(z[-1], current_int_end)
                + (current_int_end - z[-1]) * self._b[ch]
            ) / self._kappa[ch]
            if len(z) == 1:
                si = si + (self._integrator_init[ch] + self._delta[ch])
            if np.abs(si - prvs_integral) / np.abs(si) < tolerance:
                z.append(current_int_end)
                low_int_bound = current_int_end
                upp_int_bound = signal_end_time
                current_int_end = signal_end_time
            elif si > 2 * self._delta[ch]:
                upp_int_bound = current_int_end
                current_int_end = (low_int_bound + current_int_end) / 2
            else:
                low_int_bound = current_int_end
                current_int_end = (current_int_end + upp_int_bound) / 2
            if (
                signal.get_precise_integral(z[-1], signal_end_time)
                + (signal_end_time - z[-1]) * self._b[ch]
            ) / self._kappa[ch] < 2 * self._delta[ch]:
                break
            prvs_integral = si

        return z if with_start_time else z[1:]

    def encode(
        self,
        x_param,
        signal_end_time,
        tolerance=1e-8,
        with_start_time=False,
    ):
        """
        performs the encoding of the input signals at channel ch of the encoder
        with a spike time output that is generated according to an integrate-
        and-fire time encoding scheme

        PARAMETERS
        ----------
        x_param: Signal.SignalCollection
            signal(s) to be encoded
        signal_end_time: float
            time at which encoding should stop
        tolerance: float
            tolerated error on the spike timess
        with_start_time: bool
            determines whether or not the starting time of the time encoding
            is included as the first spike time

        RETURNS
        -------
        SpikeTimes
            spikeTime object holding the timing of the spikes of each of
            the channels
        """

        self.__dict__.update(self.params.__dict__)

        y_param = x_param.get_mixed_signals(self.mixing_matrix)
        spikes = SpikeTimes(self.n_channels)
        for ch in range(self.n_channels):
            signal = y_param[ch]
            spikes_of_ch = self.encode_single_channel_precise(
                signal,
                signal_end_time,
                ch,
                tolerance,
                with_start_time,
            )
            spikes.add(ch, spikes_of_ch)
        return spikes
