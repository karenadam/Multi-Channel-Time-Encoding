import numpy as np
from scipy.special import sici
import numpy.matlib
import bisect
import copy
from Spike_Times import spikeTimes


class timeEncoder(object):
    def __init__(
        self,
        kappa,
        delta,
        b,
        n_channels=1,
        integrator_init=[0],
        tol=1e-12,
        precision=10,
    ):
        if isinstance(delta, (list)):
            self.precision = int(precision + 1 / max(delta))
        else:
            self.precision = int(precision + 1 / (delta))
        self.n_channels = n_channels
        self.kappa = self.check_dimensions(kappa)
        self.delta = self.check_dimensions(delta)
        self.integrator_init = self.check_dimensions(integrator_init)
        self.b = self.check_dimensions(b)
        self.tol = tol

    
    def set_b(self, b):
        self.b = self.check_dimensions(b)

    def check_dimensions(self, parameter):
        if not isinstance(parameter, (list)):
            parameter = [parameter] * self.n_channels
        elif len(parameter) == 1:
            parameter = parameter * self.n_channels
        else:
            assert (
                len(parameter) == self.n_channels
            ), "There should be as many values set for the TEM parameters as there are channels"
            parameter = parameter
        return parameter

    def encode(self, signal, delta_t, with_integral_probe=False):
        spikes = spikeTimes(self.n_channels)
        if with_integral_probe:
            integrator_output = np.zeros((self.n_channels, max(signal.shape)))
        for ch in range(self.n_channels):
            spike_locations = []
            run_sum = np.cumsum(delta_t * (signal + self.b[ch])) / self.kappa[ch]
            integrator = self.integrator_init[ch]
            thresh = self.delta[ch] - integrator
            nextpos = bisect.bisect_left(run_sum, thresh)
            while nextpos != len(signal):
                spike_locations.append(nextpos)
                spikes.add(ch, nextpos * delta_t)
                thresh = thresh + 2 * self.delta[ch]
                nextpos = bisect.bisect_left(run_sum, thresh)
            if with_integral_probe:
                run_sum += self.integrator_init[ch]
                for spike_loc in spike_locations:
                    run_sum[spike_loc:] -= 2 * self.delta[ch]
                integrator_output[ch, :] = run_sum[:]

        if with_integral_probe:
            return spikes, integrator_output
        else:
            return spikes

    def compute_integral(self, sinc_loc, sinc_amp, Omega, start_time, end_time, b):
        integral = b * (end_time - start_time)
        for l in range(len(sinc_loc)):
            integral += (
                sinc_amp[l]
                * (
                    sici(Omega * (end_time - sinc_loc[l]))[0]
                    - sici(Omega * (start_time - sinc_loc[l]))[0]
                )
                / np.pi
            )
        return integral

    def encode_single_channel_precise(
        self,
        sinc_loc,
        sinc_amp,
        Omega,
        signal_end_time,
        channel=0,
        tolerance=1e-6,
        ic_integrator_default=True,
        ic_integrator=0,
    ):
        if ic_integrator_default:
            integrator = -self.delta[channel]
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
                self.compute_integral(
                    sinc_loc, sinc_amp, Omega, z[-1], current_int_end, self.b[channel]
                )
                / self.kappa[channel]
            )
            if len(z) == 1:
                si = si + (integrator + self.delta[channel])
                # si += integrator
            if np.abs(si - prvs_integral) < tolerance:
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
                self.compute_integral(
                    sinc_loc, sinc_amp, Omega, z[-1], signal_end_time, self.b[channel]
                )
                / self.kappa[channel]
                < 2 * self.delta[channel]
            ):
                break
            prvs_integral = si
        return z[1:]

    def encode_precise(self, sinc_loc, sinc_amp, Omega, signal_end_time, tol=1e-8):
        spikes = spikeTimes(self.n_channels)
        for ch in range(self.n_channels):
            spikes_of_n = self.encode_single_channel_precise(
                sinc_loc,
                sinc_amp,
                Omega,
                signal_end_time,
                channel=ch,
                tolerance=tol,
                ic_integrator_default=False,
                ic_integrator=self.integrator_init[ch],
            )
            spikes.add(ch, spikes_of_n)
        return spikes

    def decode(self, spikes, t, Omega, Delta_t):
        x = np.zeros_like(t)
        q, G = self.get_closed_form_matrices(spikes, Omega)
        G_pl = np.linalg.pinv(G, rcond=1e-15)

        start_index = 0
        for ch in range(self.n_channels):
            n_spikes_in_ch = spikes.get_n_spikes_of(ch)
            spike_midpoints = spikes.get_midpoints(ch)
            for l in range(n_spikes_in_ch - 1):
                x += (
                    G_pl[start_index + l, :].dot(q)
                    * np.sinc(Omega * (t - spike_midpoints[l]) / np.pi)
                    * Omega
                    / np.pi
                )
            start_index += n_spikes_in_ch - 1

        return x

    def get_closed_form_matrices(self, spikes, Omega):
        n_spikes = spikes.get_total_num_spikes()
        q = np.zeros((n_spikes - self.n_channels, 1))
        G = np.zeros((n_spikes - self.n_channels, n_spikes - self.n_channels))

        start_index = 0
        for ch in range(self.n_channels):
            n_spikes_in_ch = spikes.get_n_spikes_of(ch)
            spikes_in_ch = spikes.get_spikes_of(ch)
            spike_diff = spikes_in_ch[1:] - spikes_in_ch[:-1]
            q[start_index : start_index + n_spikes_in_ch - 1] = np.transpose(
                np.atleast_2d(
                    (-self.b[ch] * (spike_diff) + 2 * self.kappa[ch] * (self.delta[ch]))
                )
            )

            start_index_j = 0
            for ch_j in range(self.n_channels):
                n_spikes_in_ch_j = spikes.get_n_spikes_of(ch_j)
                spike_midpoints_j = spikes.get_midpoints(ch_j)
                up_bound = np.transpose(
                    np.matlib.repmat(spikes_in_ch[1:], n_spikes_in_ch_j - 1, 1)
                ) - np.matlib.repmat(spike_midpoints_j, n_spikes_in_ch - 1, 1)
                low_bound = np.transpose(
                    np.matlib.repmat(spikes_in_ch[:-1], n_spikes_in_ch_j - 1, 1)
                ) - np.matlib.repmat(spike_midpoints_j, n_spikes_in_ch - 1, 1)

                G[
                    start_index : start_index + n_spikes_in_ch - 1,
                    start_index_j : start_index_j + n_spikes_in_ch_j - 1,
                ] = (sici(Omega * up_bound)[0] - sici(Omega * low_bound)[0]) / np.pi
                start_index_j += n_spikes_in_ch_j - 1

            start_index += n_spikes_in_ch - 1

        return q, G
