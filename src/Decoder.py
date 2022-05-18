from src import *


class Decoder(object):
    def __init__(self, tem_parameters):
        self.params = tem_parameters
        self.__dict__.update(self.params.__dict__)

    def decode(self, signal):
        raise NotImplementedError

    def check_signal_type(self, periodic, Omega, period, n_components):
        assert (not periodic and Omega is not None) or (
            periodic and (period is not None) and (n_components is not None)
        ), "the type of signal is not consistent with the parameters given"

    def apply_kernels(
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

        def Ki(t):
            return (
                Helpers.Di(t, period, n_components)
                if periodic
                else Helpers.Si(t, Omega)
            )

        kernels = np.concatenate(
            [
                self.get_kernels_from_spikes(Ki, spikes, ch, t)
                for ch in range(self.n_channels)
            ]
        )
        x = G_pl.dot(q).dot(kernels)
        return x

    def get_kernels_from_spikes(self, Ki, spikes, ch, t):
        spikes_in_ch = np.atleast_2d(spikes.get_spikes_of(ch)).T
        t = np.atleast_2d(t)
        kernels = (Ki(t - spikes_in_ch[1:, :]) - Ki(t - spikes_in_ch[:-1, :])) / (
            np.diff(spikes_in_ch, axis=0)
        )
        return kernels

    def unweighted_multi_channel(self):
        # Checks if one signal is fed with weight one to all channels
        # If it is, the reconstruction can be done in closed form
        if self.mixing_matrix.shape[1] == 1:
            if (self.mixing_matrix == np.ones_like(self.mixing_matrix)).all():
                return True
        return False

    def get_measurement_vector(self, spikes):
        q = np.concatenate(
            [
                -self.b[ch] * (np.diff(spikes.get_spikes_of(ch), axis=0))
                + 2 * self.kappa[ch] * (self.delta[ch])
                for ch in range(self.n_channels)
            ]
        )
        return q

    def get_measurement_bloc(self, spikes, Kii, ch_i, ch_j):
        (
            sum_k_l,
            sum_k1_l1,
            sum_k1_l,
            sum_k_l1,
            diff_l1_l,
        ) = self.get_integral_start_and_end_points(
            spikes.get_spikes_of(ch_i), spikes.get_spikes_of(ch_j)
        )

        return (Kii(sum_k1_l1) - Kii(sum_k_l1) - Kii(sum_k1_l) + Kii(sum_k_l)) / (
            diff_l1_l
        )

    def get_measurement_operator(
        self, spikes, periodic=False, Omega=None, period=None, n_components=None
    ):
        self.check_signal_type(periodic, Omega, period, n_components)

        def Kii(t):
            return (
                Helpers.Dii(t, period, n_components)
                if periodic
                else Helpers.Sii(t, Omega)
            )

        return np.concatenate(
            [
                np.concatenate(
                    [
                        self.get_measurement_bloc(spikes, Kii, ch, ch_j)
                        for ch_j in range(self.n_channels)
                    ],
                    axis=1,
                )
                for ch in range(self.n_channels)
            ],
            axis=0,
        )

    def get_integral_start_and_end_points(self, spikes_in_ch_i, spikes_in_ch_j):
        t_k_matrix = np.transpose(
            np.matlib.repmat(spikes_in_ch_i, len(spikes_in_ch_j), 1)
        )
        t_l_matrix = np.matlib.repmat(spikes_in_ch_j, len(spikes_in_ch_i), 1)

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
        self.check_signal_type(periodic, Omega, period, n_components)

        q = self.get_measurement_vector(spikes)
        G = self.get_measurement_operator(
            spikes,
            periodic=periodic,
            Omega=Omega,
            period=period,
            n_components=n_components,
        )
        G_pl = np.linalg.pinv(G, rcond=cond_n)

        x = self.apply_kernels(
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
    def get_PCS_sampler_from_spikes(self, spikes, sinc_locs, Omega):
        discontinuities = [
            spikes.get_spikes_of(ch).tolist() for ch in range(self.n_channels)
        ]
        values = [[0] * (len(discontinuities[ch]) - 1) for ch in range(self.n_channels)]

        PCSSignal = Signal.piecewiseConstantSignals(discontinuities, values)

        PCS_sampler_normalizer = np.concatenate(
            [np.diff(spikes.get_spikes_of(ch)) for ch in range(self.n_channels)]
        )
        PCS_sampler = (
            PCSSignal.get_sampler_matrix(sinc_locs, Omega) / PCS_sampler_normalizer
        )

        return PCS_sampler

    def get_measurement_operators_sincs(self, spikes, sinc_locs, Omega):
        mixing_matrix_inv = np.linalg.pinv(self.mixing_matrix)

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

        integral_computing_matrix = Ysincs.get_integral_matrix(t_start, t_end)
        flattened_backward_mixing = Ysincs.get_flattened_mixing_matrix(
            mixing_matrix_inv
        )
        flattened_forward_mixing = Xsincs.get_flattened_mixing_matrix(
            self.mixing_matrix
        )
        PCS_sampler = self.get_PCS_sampler_from_spikes(spikes, sinc_locs, Omega)

        return (
            integral_computing_matrix,
            flattened_backward_mixing,
            flattened_forward_mixing,
            PCS_sampler,
        )

    def decode(self, spikes, t, sinc_locs, Omega, Delta_t, return_as_param=False):
        self.__dict__.update(self.params.__dict__)

        q = self.get_measurement_vector(spikes)
        (
            integral_computing_matrix,
            flattened_backward_mixing,
            flattened_forward_mixing,
            PCS_sampler,
        ) = self.get_measurement_operators_sincs(spikes, sinc_locs, Omega)

        operator_inverse = np.linalg.pinv(
            flattened_backward_mixing.dot(PCS_sampler)
            .dot(integral_computing_matrix)
            .dot(flattened_forward_mixing)
        )

        x_sinc_amps = (
            operator_inverse.dot(flattened_backward_mixing).dot(PCS_sampler).dot(q)
        ).reshape((self.n_signals, len(sinc_locs)))

        x_param = Signal.bandlimitedSignals(Omega, sinc_locs, x_sinc_amps)

        return x_param if return_as_param else x_param.sample(t)

    def get_measurement_operator_periodic(self, spikes, period, n_components):

        FS_components = (
            1j * 2 * np.pi / period * np.arange(-n_components + 1, n_components, 1)
        )

        def get_measurement_operator_bloc(ch):
            spikes_in_ch = spikes.get_spikes_of(ch)
            a_ch = np.atleast_2d(self.mixing_matrix[ch, :])
            integrals = Helpers.exp_int(
                FS_components, spikes_in_ch[:-1:], spikes_in_ch[1::]
            )
            return np.real(
                np.transpose(np.multiply.outer(a_ch, integrals), (3, 1, 2, 0)).reshape(
                    (len(spikes_in_ch) - 1, -1)
                )
            )

        measurement_operator = np.concatenate(
            [get_measurement_operator_bloc(ch) for ch in range(self.n_channels)], axis=0
        )

        return measurement_operator

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

        assert (
            periodic and (period is not None) and (n_components is not None)
        ), "the type of signal is not consistent with the parameters given"

        measurement_vector = self.get_measurement_vector(spikes)
        measurement_operator = self.get_measurement_operator_periodic(
            spikes, period, n_components
        )

        recovered_coefficients = (
            np.linalg.pinv(measurement_operator).dot(measurement_vector)
        ).reshape((self.n_signals, 2 * n_components - 1))

        x_param = Signal.periodicBandlimitedSignals(
            period, n_components, recovered_coefficients
        )

        return x_param if return_as_param else x_param.sample(t)

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
