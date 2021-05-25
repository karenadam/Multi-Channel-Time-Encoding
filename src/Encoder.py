from src import *


class Encoder(object):
    def __init__(self, tem_parameters, with_integral_probe=False):
        self.params = tem_parameters
        self.with_integral_probe = with_integral_probe

    def encode(self, signal):
        raise NotImplementedError


class DiscreteEncoder(Encoder):
    def encode(self, signal, signal_end_time, delta_t):
        self.__dict__.update(self.params.__dict__)

        spikes = spikeTimes(self.n_channels)
        if isinstance(signal, Signal.Signal) or isinstance(
            signal, Signal.SignalCollection
        ):
            sampled = signal.sample(np.arange(0, signal_end_time, delta_t))
        else:
            sampled = signal
        input_signal = self.mix_signals(sampled)
        if self.with_integral_probe:
            integrator_output = np.zeros((self.n_channels, max(signal.shape)))
        for ch in range(self.n_channels):
            spike_locations = []
            input_to_ch = input_signal[ch, :]
            run_sum = np.cumsum(delta_t * (input_to_ch + self.b[ch])) / self.kappa[ch]
            integrator = self.integrator_init[ch]
            thresh = self.delta[ch] - integrator
            nextpos = bisect.bisect_left(run_sum, thresh)
            while nextpos != len(input_to_ch):
                spike_locations.append(nextpos)
                spikes.add(ch, nextpos * delta_t)
                thresh = thresh + 2 * self.delta[ch]
                nextpos = bisect.bisect_left(run_sum, thresh)
            if self.with_integral_probe:
                run_sum += self.integrator_init[ch]
                for spike_loc in spike_locations:
                    run_sum[spike_loc:] -= 2 * self.delta[ch]
                integrator_output[ch, :] = run_sum[:]

        if self.with_integral_probe:
            return spikes, integrator_output
        else:
            return spikes

    def mix_signals(self, signal):
        print("bla", self.mixing_matrix.shape)
        print("bli", signal.shape)
        signal = np.atleast_2d(signal)
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T
        assert len(signal.shape) == 2, "Your signals should have 2 dimensions"
        assert (
            self.mixing_matrix.shape[1] == signal.shape[0]
        ), "Your signals and your mixing matrix have mismatching dimensions"
        input_signal = self.mixing_matrix.dot(signal)
        return input_signal


class ContinuousEncoder(Encoder):
    def compute_integral(self, signal, start_time, end_time, b=0):
        integral = signal.get_precise_integral(start_time, end_time) + b * (
            end_time - start_time
        )
        return integral

    def encode_single_channel_precise(
        self,
        signal,
        signal_end_time,
        channel=0,
        tolerance=1e-6,
        ic_integrator_default=True,
        ic_integrator=0,
        with_start_time=False,
    ):
        def get_int(a, b):
            return signal.get_precise_integral(a, b)

        if ic_integrator_default:
            integrator = -self.integrator_init[channel]
        else:
            integrator = ic_integrator
        z = [0]
        prvs_integral = 0
        current_int_end = signal_end_time
        upp_int_bound = signal_end_time
        low_int_bound = 0
        if signal_end_time == 0:
            return []
        counter = 0
        while z[-1] < signal_end_time:
            si = (
                get_int(z[-1], current_int_end)
                + (current_int_end - z[-1]) * self.b[channel]
            ) / self.kappa[channel]
            # print(si)
            if len(z) == 1:
                si = si + (integrator + self.delta[channel])
            if np.abs(si - prvs_integral) / np.abs(si) < tolerance:
                if len(z) > 1 and z[-1] - z[-2] < tolerance:
                    z = z[:-1]
                z.append(current_int_end)
                low_int_bound = current_int_end
                upp_int_bound = signal_end_time
                current_int_end = signal_end_time
            elif si > 2 * self.delta[channel]:
                upp_int_bound = current_int_end
                current_int_end = (low_int_bound + current_int_end) / 2
            else:
                low_int_bound = current_int_end
                current_int_end = (current_int_end + upp_int_bound) / 2
            if (
                get_int(z[-1], signal_end_time)
                + (signal_end_time - z[-1]) * self.b[channel]
            ) / self.kappa[channel] < 2 * self.delta[channel]:
                break
            prvs_integral = si

        if with_start_time:
            return z
        else:
            return z[1:]

    def encode(
        self,
        x_param,
        signal_end_time,
        tol=1e-8,
        same_sinc_locs=True,
        with_start_time=False,
    ):

        self.__dict__.update(self.params.__dict__)
        assert isinstance(x_param, Signal.bandlimitedSignals) or isinstance(
            x_param, Signal.periodicBandlimitedSignals
        )
        n_signals = x_param.n_signals

        y_param = x_param.get_mixed_signals(self.mixing_matrix)

        spikes = spikeTimes(self.n_channels)
        for ch in range(self.n_channels):
            # signal = bandlimitedSignal(Omega, x_sinc_locs, y_sinc_amps[ch])
            signal = y_param.get_signal(ch)

            spikes_of_ch = self.encode_single_channel_precise(
                signal,
                signal_end_time,
                channel=ch,
                tolerance=tol,
                ic_integrator_default=False,
                ic_integrator=self.integrator_init[ch],
                with_start_time=with_start_time,
            )
            spikes.add(ch, spikes_of_ch)
        return spikes
