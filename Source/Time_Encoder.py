import numpy as np
from scipy.special import sici
import numpy.matlib
import bisect
import copy
from Signal import *
from Spike_Times import spikeTimes


class timeEncoder(object):
    def __init__(
        self,
        kappa,
        delta,
        b,
        mixing_matrix,
        integrator_init=[0],
        tol=1e-12,
        precision=10,
    ):
        self.mixing_matrix = np.atleast_2d(np.array(mixing_matrix))
        self.n_signals = self.mixing_matrix.shape[1]
        self.n_channels = self.mixing_matrix.shape[0]
        if isinstance(delta, (list)):
            self.precision = int(precision + 1 / max(delta))
        else:
            self.precision = int(precision + 1 / (delta))
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
        input_signal = self.mix_signals(signal)
        if with_integral_probe:
            integrator_output = np.zeros((self.n_channels, max(signal.shape)))
        for ch in range(self.n_channels):
            input_to_ch = input_signal[ch, :]
            run_sum = np.cumsum(delta_t * (input_to_ch + self.b[ch])) / self.kappa[ch]
            integrator = self.integrator_init[ch]
            thresh = self.delta[ch] - integrator
            nextpos = bisect.bisect_left(run_sum, thresh)
            while nextpos != len(input_to_ch):
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

    def mix_signals(self, signal):
        signal = np.atleast_2d(signal)
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T
        assert len(signal.shape) == 2, "Your signals should have 2 dimensions"
        assert (
            self.mixing_matrix.shape[1] == signal.shape[0]
        ), "Your signals and your mixing matrix have mismatching dimensions"
        input_signal = self.mixing_matrix.dot(signal)
        return input_signal

    def compute_integral(self, sinc_loc, sinc_amp, Omega, start_time, end_time, b=0):
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

    def encode_precise(
        self,
        x_param,
        Omega,
        signal_end_time,
        tol=1e-8,
        same_sinc_locs=True,
    ):
        if isinstance(x_param, list):
            n_signals = len(x_param)
        else:
            n_signals = 1
        if same_sinc_locs:
            x_sinc_locs = []
            x_sinc_amps = []
            for n in range(n_signals):
                if isinstance(x_param, list):
                    sinc_locs, sinc_amps = x_param[n].get_sincs()
                else:
                    sinc_locs, sinc_amps = x_param.get_sincs()
                x_sinc_amps.append(sinc_amps)
                if n == 0:
                    x_sinc_locs = sinc_locs
            y_sinc_amps = self.mixing_matrix.dot(np.array(x_sinc_amps))

        spikes = spikeTimes(self.n_channels)
        for ch in range(self.n_channels):
            spikes_of_ch = self.encode_single_channel_precise(
                x_sinc_locs,
                y_sinc_amps[ch],
                Omega,
                signal_end_time,
                channel=ch,
                tolerance=tol,
                ic_integrator_default=False,
                ic_integrator=self.integrator_init[ch],
            )
            spikes.add(ch, spikes_of_ch)
        return spikes

    def decode(self, spikes, t, Omega, Delta_t, cond_n=1e-15):      
        y = np.zeros((self.n_signals, len(t)))   
        q, G = self.get_closed_form_matrices2(spikes, Omega)
        G_pl = np.linalg.pinv(G, rcond=cond_n)

        x = self.apply_g(G_pl, q, spikes, t, Omega)


        return x  


    def get_closed_form_matrices(self, spikes, Omega):
        n_spikes = spikes.get_total_num_spikes()

        q = np.zeros((n_spikes - self.n_channels, self.n_channels))
        G = np.zeros((n_spikes - self.n_channels, n_spikes - self.n_channels))

        start_index = 0
        for ch in range(self.n_channels):
            n_spikes_in_ch = spikes.get_n_spikes_of(ch)
            spikes_in_ch = spikes.get_spikes_of(ch)
            spike_diff = spikes_in_ch[1:] - spikes_in_ch[:-1]
            q[start_index : start_index + n_spikes_in_ch - 1, ch] = -self.b[ch] * (
                    spike_diff
                    ) + 2 * self.kappa[ch] * (self.delta[ch])

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

        if self.unweighted_multi_channel():
            q = np.sum(q,1)

        return q, G

    def get_closed_form_matrices2(self, spikes, Omega):
        n_spikes = spikes.get_total_num_spikes()

        q = np.zeros((n_spikes - self.n_channels, self.n_channels))
        G = np.zeros((n_spikes - self.n_channels, n_spikes - self.n_channels))

        start_index = 0
        for ch in range(self.n_channels):
            n_spikes_in_ch = spikes.get_n_spikes_of(ch)
            spikes_in_ch = spikes.get_spikes_of(ch)
            spike_diff = spikes_in_ch[1:] - spikes_in_ch[:-1]
            q[start_index : start_index + n_spikes_in_ch - 1, ch] = -self.b[ch] * (
                    spike_diff
                    ) + 2 * self.kappa[ch] * (self.delta[ch])

            start_index_j = 0
            for ch_j in range(self.n_channels):
                n_spikes_in_ch_j = spikes.get_n_spikes_of(ch_j)
                spikes_in_ch_j = spikes.get_spikes_of(ch_j)
                spike_midpoints_j = spikes.get_midpoints(ch_j)

                t_k_matrix = np.transpose(np.matlib.repmat(spikes_in_ch, n_spikes_in_ch_j, 1))
                t_l_matrix = np.matlib.repmat(spikes_in_ch_j, n_spikes_in_ch, 1)

                sum_k_l = t_k_matrix[:-1,:-1]-t_l_matrix[:-1,:-1]
                sum_k1_l1 = t_k_matrix[1:,1:]-t_l_matrix[1:,1:]
                sum_k1_l = t_k_matrix[1:,1:]-t_l_matrix[:-1,:-1]
                sum_k_l1 = t_k_matrix[:-1,:-1]-t_l_matrix[1:,1:]

                G[
                    start_index : start_index + n_spikes_in_ch - 1,
                    start_index_j : start_index_j + n_spikes_in_ch_j - 1,
                ] = np.cos(sum_k1_l1)-np.cos(sum_k_l1)-np.cos(sum_k1_l)+np.cos(sum_k_l) +\
                sum_k1_l1*sici(sum_k1_l1)[0] - sum_k_l1*sici(sum_k_l1)[0] - sum_k1_l*sici(sum_k1_l)[0]+\
                sum_k_l*sici(sum_k_l)[0]


                up_bound = np.transpose(
                    np.matlib.repmat(spikes_in_ch[1:], n_spikes_in_ch_j - 1, 1)
                ) - np.matlib.repmat(spike_midpoints_j, n_spikes_in_ch - 1, 1)
                low_bound = np.transpose(
                    np.matlib.repmat(spikes_in_ch[:-1], n_spikes_in_ch_j - 1, 1)
                ) - np.matlib.repmat(spike_midpoints_j, n_spikes_in_ch - 1, 1)

                # G[
                #     start_index : start_index + n_spikes_in_ch - 1,
                #     start_index_j : start_index_j + n_spikes_in_ch_j - 1,
                # ] = (sici(Omega * up_bound)[0] - sici(Omega * low_bound)[0]) / np.pi
                start_index_j += n_spikes_in_ch_j - 1

            start_index += n_spikes_in_ch - 1

        if self.unweighted_multi_channel():
            q = np.sum(q,1)

        return q, G

    def unweighted_multi_channel(self):
        # Checks if one signal is fed with weight one to all channels
        # If it is, the reconstruction can be done in closed form
        if self.mixing_matrix.shape[1] == 1:
            if (self.mixing_matrix == np.ones_like(self.mixing_matrix)).all():
                return True
        return False

    def decode_recursive(
        self, spikes, t, Omega, Delta_t, num_iterations=1
    ):

        q, G = self.get_closed_form_matrices(
            spikes,Omega
        )
        q = q.T

        mixing_matrix_inv = np.linalg.inv(self.mixing_matrix.T.dot(self.mixing_matrix)).dot(
            self.mixing_matrix.T
        )

        sinc_locs = spikes.get_all_midpoints()

        q_x = mixing_matrix_inv.dot(q)
        x_param_0 = []
        x_param_0 = bandlimitedSignals(
            t, Delta_t, Omega, sinc_locs=sinc_locs, sinc_amps=q_x
        )

        q_y = self.mixing_matrix.dot(q_x)
        y_param = bandlimitedSignals(
            t, Delta_t, Omega, sinc_locs=sinc_locs, sinc_amps=q_y
        )
        x_param = copy.deepcopy(x_param_0)
        for n_iter in range(num_iterations):
            q_l = self.get_integrals_param(y_param.get_signals(), spikes)
            rec_matrix = np.atleast_2d(mixing_matrix_inv.dot(q_l))
            for n in range(self.n_signals):
                sinc_amps_l = x_param.get_signals()[n].get_sinc_amps()
                sinc_amps_0 = x_param_0.get_signals()[n].get_sinc_amps()
                x_param.get_signals()[n].set_sinc_amps(
                    [
                        sinc_amps_l[k] + sinc_amps_0[k] - rec_matrix[n, k]
                        for k in range(len(sinc_locs))
                    ]
                )
            y_sinc_amps = self.mix_sinc_amps(x_param.get_signals(), self.mixing_matrix)
            y_param = bandlimitedSignals(
                t, Delta_t, Omega, sinc_locs=sinc_locs, sinc_amps=y_sinc_amps
            )
        return x_param.sample(t)

    def mix_sinc_amps(self, x_param, mixing_matrix):
        num_x = len(x_param)
        x_sinc_amps = np.zeros((num_x, len(x_param[0].get_sinc_amps())))
        for n in range(num_x):
            x_sinc_amps[n, :] = x_param[n].get_sinc_amps()
        y_sinc_amps = mixing_matrix.dot(x_sinc_amps)
        return y_sinc_amps

    def get_integrals_param(self, sig_param, spikes):
        q_shape = (len(sig_param), spikes.get_total_num_constraints())
        q = np.zeros(q_shape)
        start_index = 0
        for ch in range(self.n_channels):
            sig = sig_param[ch]
            Omega = sig.Omega
            sig_sinc_locs, sig_sinc_amps = sig.get_sincs()
            spike_times = spikes.get_spikes_of(ch)
            for i in range(len(spike_times) - 1):
                q[ch, start_index + i] = self.compute_integral(
                    sig_sinc_locs,
                    sig_sinc_amps,
                    Omega,
                    spike_times[i],
                    spike_times[i + 1],
                )
            start_index += len(spike_times) - 1
        return q

    def get_integrals(self, signal, spikes, Delta_t, q_shape):
        q = np.zeros(q_shape)
        start_index = 0
        for ch in range(self.n_channels):
            spike_times = spikes.get_spikes_of(ch)
            spike_indices = [int(t / Delta_t) for t in spike_times]
            signal_cum_sum = np.cumsum(Delta_t * signal[ch, :])
            for i in range(len(spike_times) - 1):
                q[start_index + i, ch] = (
                    signal_cum_sum[spike_indices[i + 1]]
                    - signal_cum_sum[spike_indices[i]]
                )
            start_index += len(spike_times) - 1
        return q

    def apply_g(self, G_pl, q, spikes, t, Omega):
        x = np.zeros_like(t)   
        start_index = 0
        for ch in range(self.n_channels):
            n_spikes_in_ch = spikes.get_n_spikes_of(ch)
            spikes_in_ch = spikes.get_spikes_of(ch)
            spike_midpoints = spikes.get_midpoints(ch)

            sici_upp_in = (np.atleast_2d(t) - np.atleast_2d(spikes_in_ch[1:]).T)/np.pi
            sici_low_in = (np.atleast_2d(t) - np.atleast_2d(spikes_in_ch[:-1]).T)/np.pi
            kernel = Omega/np.pi* (sici(sici_upp_in)[0] - sici(sici_low_in)[0])/(spikes_in_ch[1:,None]- spikes_in_ch[:-1,None])

            x += (G_pl[start_index:start_index+n_spikes_in_ch-1].dot(q).dot(kernel))
            # for l in range(n_spikes_in_ch - 1):
            #     x += (
            #         G_pl[start_index + l, :].dot(q)
            #         * np.sinc(Omega * (t - spike_midpoints[l]) / np.pi)
            #         * Omega
            #         / np.pi
            #     )
            start_index += n_spikes_in_ch - 1
        return x  
