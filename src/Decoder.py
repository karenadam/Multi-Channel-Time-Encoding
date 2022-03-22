from src import *

class Decoder(object):
    def __init__(self, tem_parameters):
        self.params = tem_parameters
        self.__dict__.update(self.params.__dict__)

    def decode(self, signal):
        raise NotImplementedError

    def apply_g(
        self,
        G_pl,
        q,
        spikes,
        t,
        periodic=False,
        Omega=None,
        period=None,
        n_components=None,
    ):
        assert (not periodic and Omega is not None) or (
            periodic and (period is not None) and (n_components is not None)
        ), "the type of signal is not consistent with the parameters given"

        def Ki(t):
            if periodic:
                return Helpers.Di(t, period, n_components)
            else:
                return Helpers.Si(t, Omega)

        x = np.zeros_like(t)
        start_index = 0
        for ch in range(self.n_channels):
            n_spikes_in_ch = spikes.get_n_spikes_of(ch)
            spikes_in_ch = spikes.get_spikes_of(ch)
            spike_midpoints = spikes.get_midpoints(ch)

            integral_up_bound = np.atleast_2d(t) - np.atleast_2d(spikes_in_ch[1:]).T
            integral_low_bound = np.atleast_2d(t) - np.atleast_2d(spikes_in_ch[:-1]).T
            kernel = (Ki(integral_up_bound) - Ki(integral_low_bound)) / (
                spikes_in_ch[1:, None] - spikes_in_ch[:-1, None]
            )

            x += G_pl[start_index : start_index + n_spikes_in_ch - 1].dot(q).dot(kernel)
            start_index += n_spikes_in_ch - 1
        return x

    def unweighted_multi_channel(self):
        # Checks if one signal is fed with weight one to all channels
        # If it is, the reconstruction can be done in closed form
        if self.mixing_matrix.shape[1] == 1:
            if (self.mixing_matrix == np.ones_like(self.mixing_matrix)).all():
                return True
        return False

    def get_closed_form_matrices(
        self, spikes, periodic=False, Omega=None, period=None, n_components=None
    ):
        assert (not periodic and Omega is not None) or (
            periodic and (period is not None) and (n_components is not None)
        ), "the type of signal is not consistent with the parameters given"
        n_spikes = spikes.get_total_num_spikes()

        def Kii(t):
            if periodic:
                return Helpers.Dii(t, period, n_components)
            else:
                return Helpers.Sii(t, Omega)

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

                sum_k_l, sum_k1_l1, sum_k1_l, sum_k_l1, diff_l1_l = self.get_integral_start_and_end_points(spikes_in_ch, spikes_in_ch_j)

                G[
                    start_index : start_index + n_spikes_in_ch - 1,
                    start_index_j : start_index_j + n_spikes_in_ch_j - 1,
                ] = (Kii(sum_k1_l1) - Kii(sum_k_l1) - Kii(sum_k1_l) + Kii(sum_k_l)) / (
                    diff_l1_l
                )

                start_index_j += n_spikes_in_ch_j - 1

            start_index += n_spikes_in_ch - 1

        if self.unweighted_multi_channel():
            q = np.sum(q, 1)

        return q, G

    def get_integral_start_and_end_points(self, spikes_in_ch, spikes_in_ch_j):
        n_spikes_in_ch_j = len(spikes_in_ch_j)
        n_spikes_in_ch = len(spikes_in_ch)

        t_k_matrix = np.transpose(
            np.matlib.repmat(spikes_in_ch, n_spikes_in_ch_j, 1)
        )
        t_l_matrix = np.matlib.repmat(spikes_in_ch_j, n_spikes_in_ch, 1)
        sum_k_l = t_k_matrix[:-1, :-1] - t_l_matrix[:-1, :-1]
        sum_k1_l1 = t_k_matrix[1:, 1:] - t_l_matrix[1:, 1:]
        sum_k1_l = t_k_matrix[1:, 1:] - t_l_matrix[:-1, :-1]
        sum_k_l1 = t_k_matrix[:-1, :-1] - t_l_matrix[1:, 1:]
        diff_l1_l = t_l_matrix[1:, 1:] - t_l_matrix[:-1, :-1]
        return sum_k_l, sum_k1_l1, sum_k1_l, sum_k_l1, diff_l1_l


class SSignalMChannelDecoder(Decoder):
    def decode(
        self,
        spikes,
        t,
        periodic=False,
        Omega=None,
        period=None,
        n_components=None,
        cond_n=1e-15,
    ):
        self.__dict__.update(self.params.__dict__)
        assert (not periodic and Omega is not None) or (
            periodic and (period is not None) and (n_components is not None)
        ), "the type of signal is not consistent with the parameters given"

        y = np.zeros((self.n_signals, len(t)))
        q, G = self.get_closed_form_matrices(
            spikes,
            periodic=periodic,
            Omega=Omega,
            period=period,
            n_components=n_components,
        )
        G_pl = np.linalg.pinv(G, rcond=cond_n)

        x = self.apply_g(
            G_pl,
            q,
            spikes,
            t,
            periodic=periodic,
            Omega=Omega,
            period=period,
            n_components=n_components,
        )

        return x


class MSignalMChannelDecoder(Decoder):
    def adjust_weight(self, PCS_sampler, t_start_flattened, t_end_flattened):
        for n in range(PCS_sampler.shape[0]):
            PCS_sampler[n, :] = PCS_sampler[n, :] / (
                t_end_flattened - t_start_flattened
            )
        return PCS_sampler

    def decode(self, spikes, t, sinc_locs, Omega, Delta_t, return_as_param=False):
        self.__dict__.update(self.params.__dict__)

        q, G = self.get_closed_form_matrices(spikes, periodic=False, Omega=Omega)
        q = np.atleast_2d(q.T)
        q = np.sum(q, 0)

        mixing_matrix_inv = np.linalg.inv(
            self.mixing_matrix.T.dot(self.mixing_matrix)
        ).dot(self.mixing_matrix.T)

        mixing_projector = self.mixing_matrix.dot(mixing_matrix_inv)

        discontinuities = [
            spikes.get_spikes_of(ch).tolist() for ch in range(self.n_channels)
        ]
        values = [[0] * (len(discontinuities[ch]) - 1) for ch in range(self.n_channels)]

        PCSSignal = Signal.piecewiseConstantSignals(discontinuities, values)
        Ysincs = Signal.bandlimitedSignals(
            Omega, sinc_locs, sinc_amps=[[0] * len(sinc_locs)] * self.n_channels
        )
        Xsincs = Signal.bandlimitedSignals(
            Omega, sinc_locs, sinc_amps=[[0] * len(sinc_locs)] * self.n_signals
        )

        t_start = [
            spikes.get_spikes_of(ch).tolist()[:-1] for ch in range(self.n_channels)
        ]
        t_end = [spikes.get_spikes_of(ch).tolist()[1:] for ch in range(self.n_channels)]
        PCS_sampler = PCSSignal.get_sampler_matrix(sinc_locs, Omega)

        SumOfSincs_integ_computer = Ysincs.get_integral_matrix(t_start, t_end)

        Mback = Ysincs.get_flattened_mixing_matrix(mixing_matrix_inv)
        Mfor = Xsincs.get_flattened_mixing_matrix(self.mixing_matrix)

        t_start_flattened = np.array([item for sublist in t_start for item in sublist])
        t_end_flattened = np.array([item for sublist in t_end for item in sublist])
        PCS_sampler = self.adjust_weight(
            PCS_sampler, t_start_flattened, t_end_flattened
        )

        ps_inv = np.linalg.pinv(
            Mback.dot(PCS_sampler).dot(SumOfSincs_integ_computer).dot(Mfor)
        )

        x_sinc_amps = ps_inv.dot(Mback).dot(PCS_sampler).dot(q)

        x_sinc_amps = x_sinc_amps.reshape((self.n_signals, len(sinc_locs)))
        x_param = Signal.bandlimitedSignals(
            Omega, sinc_locs=sinc_locs, sinc_amps=x_sinc_amps
        )

        if return_as_param:
            return x_param
        else:
            return x_param.sample(t)

    def decode_periodic(
        self,
        spikes,
        t,
        periodic=False,
        Omega=None,
        period=None,
        n_components=None,
        return_as_param=False,
    ):

        assert (not periodic and Omega is not None) or (
            periodic and (period is not None) and (n_components is not None)
        ), "the type of signal is not consistent with the parameters given"
        n_spikes = spikes.get_total_num_spikes()

        q = np.zeros((n_spikes - self.n_channels, 1))
        measurement_vectors = np.zeros(
            (n_spikes - self.n_channels, self.n_signals * (2 * n_components - 1))
        )

        start_index = 0
        for ch in range(self.n_channels):
            n_spikes_in_ch = spikes.get_n_spikes_of(ch)
            spikes_in_ch = spikes.get_spikes_of(ch)
            spike_diff = spikes_in_ch[1:] - spikes_in_ch[:-1]
            q[start_index : start_index + n_spikes_in_ch - 1, 0] = -self.b[ch] * (
                spike_diff
            ) + 2 * self.kappa[ch] * (self.delta[ch])

            a_ch = np.atleast_2d(self.mixing_matrix[ch, :])
            components = (
                1j * 2 * np.pi / period * np.arange(-n_components + 1, n_components, 1)
            )
            for n in range(n_spikes_in_ch - 1):
                integrals = Helpers.exp_int(
                    components, [spikes_in_ch[n]], [spikes_in_ch[n + 1]]
                )
                integrals[n_components - 1] = spikes_in_ch[n + 1] - spikes_in_ch[n]
                measurement_vectors[start_index + n, :] = (
                    a_ch.T.dot(integrals.T)
                ).flatten()

            start_index += n_spikes_in_ch - 1

        recovered_coefficients_flattened = np.linalg.pinv(measurement_vectors).dot(q)
        recovered_coefficients = np.reshape(
            recovered_coefficients_flattened, (self.n_signals, 2 * n_components - 1)
        )

        x_param = Signal.periodicBandlimitedSignals(
            period, n_components, recovered_coefficients
        )
        if return_as_param:
            return x_param
        else:
            return x_param.sample(t)

    def get_vid_constraints(self, spikes, TEM_locations):
        n_spike_diffs = spikes.get_total_num_spike_diffs()

        q = np.zeros((n_spike_diffs, 1))
        start_coordinates = np.zeros((n_spike_diffs, 3))
        end_coordinates = np.zeros((n_spike_diffs, 3))

        start_index = 0
        for ch in range(self.n_channels):
            n_spikes_in_ch = spikes.get_n_spikes_of(ch)
            if n_spikes_in_ch <= 1:
                continue
            spikes_in_ch = spikes.get_spikes_of(ch)
            spike_diff = spikes_in_ch[1:] - spikes_in_ch[:-1]
            q[start_index : start_index + n_spikes_in_ch - 1, 0] = [
                -self.b[ch] * (sp_d) + 2 * self.kappa[ch] * (self.delta[ch])
                for sp_d in spike_diff
            ]
            start_coordinates[
                start_index : start_index + n_spikes_in_ch - 1, 0
            ] = TEM_locations[ch][0]
            start_coordinates[
                start_index : start_index + n_spikes_in_ch - 1, 1
            ] = TEM_locations[ch][1]
            start_coordinates[
                start_index : start_index + n_spikes_in_ch - 1, 2
            ] = spikes_in_ch[:-1]

            end_coordinates[
                start_index : start_index + n_spikes_in_ch - 1, 0
            ] = TEM_locations[ch][0]
            end_coordinates[
                start_index : start_index + n_spikes_in_ch - 1, 1
            ] = TEM_locations[ch][1]
            end_coordinates[
                start_index : start_index + n_spikes_in_ch - 1, 2
            ] = spikes_in_ch[1:]
            start_index += n_spikes_in_ch - 1
        return q, start_coordinates, end_coordinates


class UnknownMixingDecoder(Decoder):
    def get_matrices_unknown_mixing(
        self,
        spikes,
        periodic=False,
        Omega=None,
        sinc_locs=None,
        period=None,
        n_components=None,
    ):
        assert (not periodic and Omega is not None and sinc_locs is not None) or (
            periodic and (period is not None) and (n_components is not None)
        ), "the type of signal is not consistent with the parameters given"

        if periodic:
            n_unknowns_per_ch = n_components * 2 - 1
        else:
            n_unknowns_per_ch = len(sinc_locs)
        n_unknowns = n_unknowns_per_ch * self.n_channels

        n_spikes = spikes.get_total_num_spikes()
        n_constraints = n_spikes - self.n_channels
        q = np.zeros((n_constraints, 1))
        G = np.zeros((n_constraints, n_unknowns))

        def Ki(t):
            if periodic:
                return Helpers.Di(t, period, n_components)
            else:
                return Helpers.Si(t, Omega)

        start_index_i = 0
        start_index_j = 0
        for ch in range(self.n_channels):
            n_spikes_in_ch = spikes.get_n_spikes_of(ch)
            spikes_in_ch = spikes.get_spikes_of(ch)
            spike_diff = spikes_in_ch[1:] - spikes_in_ch[:-1]
            q[start_index_i : start_index_i + n_spikes_in_ch - 1, 0] = -self.b[ch] * (
                spike_diff
            ) + 2 * self.kappa[ch] * (self.delta[ch])

            if not periodic:
                for sp_i in range(len(spikes_in_ch) - 1):
                    G[
                        start_index_i + sp_i,
                        ch * n_unknowns_per_ch : (ch + 1) * n_unknowns_per_ch,
                    ] = Ki(spikes_in_ch[sp_i + 1] - sinc_locs) - Ki(
                        spikes_in_ch[sp_i] - sinc_locs
                    )

            else:

                components = np.arange(-n_components + 1, n_components, 1)
                for sp_i in range(len(spikes_in_ch) - 1):
                    G[
                        start_index_i + sp_i,
                        ch * n_unknowns_per_ch : (ch + 1) * n_unknowns_per_ch,
                    ] = [
                        Helpers.Di2(spikes_in_ch[sp_i + 1], period, component)
                        - Helpers.Di2(spikes_in_ch[sp_i], period, component)
                        for component in components
                    ]

            start_index_i += n_spikes_in_ch - 1
        return q, G

    def decode(
        self,
        spikes,
        t,
        rank,
        periodic=False,
        sinc_locs=None,
        Omega=None,
        period=None,
        n_components=None,
    ):
        self.__dict__.update(self.params.__dict__)

        assert (not periodic and Omega is not None and sinc_locs is not None) or (
            periodic and (period is not None) and (n_components is not None)
        ), "the type of signal is not consistent with the parameters given"

        shape = (
            self.n_channels,
            len(sinc_locs) if not periodic else 2 * n_components - 1,
        )

        q, G = self.get_matrices_unknown_mixing(
            spikes,
            periodic=periodic,
            Omega=Omega,
            sinc_locs=sinc_locs,
            period=period,
            n_components=n_components,
        )

        q = np.atleast_2d(q.T)

        G_inv = np.linalg.pinv(G)
        G_inv = G.T.dot(np.linalg.pinv(G.dot(G.T)))
        C_y = Helpers.singular_value_projection_w_matrix(
            shape, G_inv.dot(G), G_inv.dot(q.T), rank, tol=1e-3, lr=0.5
        )

        if not periodic:
            y_param = Signal.bandlimitedSignals(Omega, sinc_locs, sinc_amps=C_y)
        else:
            y_param = Signal.periodicBandlimitedSignals(period, n_components, C_y)
        return y_param.sample(t)
