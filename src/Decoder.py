from src import *


class Decoder(object):
    """
    Base class for decoders from spike times emitted by integrate-
    and-fire time encoding machines.

    ATTRIBUTES
    ----------
    params: TEMParams
        Object which holds the parameters of the time encoding machines
    n_channels: int
        counts the number of channels used to perform the encoding
    periodic: bool
        specifies if the decoder aims to recover periodic signals
    Omega: float
        bandwidth of nonperiodic signal
    period: float
        period of periodic signal
    n_components: int
        number of FS components of periodic signal
    """

    def __init__(self, tem_parameters, periodic, Omega, period, n_components):
        """
        PARAMETERS
        ----------
        tem_parameters: TEMParams
            Object which holds the parameters of the time encoding machines
        periodic: bool
            specifies if the decoder aims to recover periodic signals
        Omega: float
            bandwidth of nonperiodic signal
        period: float
            period of periodic signal
        n_components: int
            number of FS components of periodic signal
        """

        self.params = tem_parameters
        self.n_channels = self.params.n_channels
        self.check_signal_type(periodic, Omega, period, n_components)
        self.periodic = periodic
        self.Omega = Omega
        self.period = period
        self.n_components = n_components

    def __repr__(self):
        repr = "Decoder object for "
        if self.periodic:
            repr += (
                "periodic bandlimited signals with period "
                + str(self.period)
                + " and "
                + str(self.n_components)
                + " components, "
            )
        else:
            repr += (
                "nonperiodic bandlimited signals with bandwidth "
                + str(self.Omega)
                + ", "
            )
        repr += "with TEM parameters \n" + str(self.params)
        return repr

    def check_signal_type(
        self, periodic: bool, Omega: float, period: float, n_components: int
    ):
        """
        Checks that the parameters provided are consistent with the (a)periodicity
        of the signal

        Parameters
        ----------
        periodic: bool
            specifies whether or not signal to recover is periodic
        Omega: float
            specifies bandwidth of signal to recover if it is not periodic
        period: float
            specifies period of periodic signal
        n_components: int
            number of fourier series components of the periodic signal

        Raises
        ------
        ValueError
            If it is not periodic but no bandwidth is specified, or
            it is periodic and either (or both) of the period or number
            of components is not defined
        """
        if not periodic and Omega is not None:
            return
        elif periodic and (period is not None) and (n_components is not None):
            return
        else:
            raise ValueError(
                "the type of signal is not consistent with the parameters given"
            )

    def get_measurement_vector(self, spikes: SpikeTimes):
        """
        computes the results of the linear measurements imposed by the spike
        times of the integrate-and-fire time encoding machines

        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the time of the spikes emitted by different integrate-and-fire
            time encoding machines

        RETURNS
        -------
        np.ndarray
            measurement vector (resulting from measurement operator defined in
            specific decoders)
        """

        q = np.concatenate(
            [
                -self.params.b[ch] * (np.diff(spikes[ch], axis=0))
                + 2 * self.params.kappa[ch] * (self.params.delta[ch])
                for ch in range(self.n_channels)
            ]
        )
        return q


class SSignalMChannelDecoder(Decoder):
    """
    Class for a decoder that can use spike times emitted by multiple
    integrate-and-fire time encoding machines to reconstruct one signal
    fed into the multiple machine

    ATTRIBUTES
    ----------
    params: TEMParams
        Object which holds the parameters of the time encoding machines
    n_channels: int
        counts the number of channels used to perform the encoding
    periodic: bool
        specifies if the decoder aims to recover periodic signals
    Omega: float
        bandwidth of nonperiodic signal
    period: float
        period of periodic signal
    n_components: int
        number of FS components of periodic signal
    """

    def __init__(
        self, tem_parameters, periodic=False, Omega=None, period=None, n_components=None
    ):
        """
        PARAMETERS
        ----------
        tem_parameters: TEMParams
            Object which holds the parameters of the time encoding machines
        periodic: bool
            specifies if the decoder aims to recover periodic signals
        Omega: float
            bandwidth of nonperiodic signal
        period: float
            period of periodic signal
        n_components: int
            number of FS components of periodic signal
        """

        super().__init__(tem_parameters, periodic, Omega, period, n_components)

    def __repr__(self):
        repr = "Single Signal, Multi Channel Decoder object for "
        if self.periodic:
            repr += (
                "periodic bandlimited signals with period "
                + str(self.period)
                + " and "
                + str(self.n_components)
                + " components, "
            )
        else:
            repr += (
                "nonperiodic bandlimited signals with bandwidth "
                + str(self.Omega)
                + ", "
            )
        repr += "with TEM parameters \n" + str(self.params)
        return repr

    def decode(self, spikes, t, cond_n=1e-15):
        """
        Method that decodes a signal from its spike times, returning
        a sampled version of the signal by returning
        x(t) = sum_k (y_k kernel(t-s_k))
        where s_k are the midpoints of the spike times,
        y_k = pinv(G).q,
        where G is measurement matrix and q is a measurement vector
        Follows method elaborated in Adam, Scholefield and Vetterli (2020)

        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the spike times of the multiple integrate-and-fire
            time encoding machines encoding the same signal
        t: np.ndarray
            vector holding the times at which one would like the recovered signal
            to be sampled

        RETURNS
        -------
        np.ndarray
            vector containing input signal sampled at times t
        """

        self.__dict__.update(self.params.__dict__)

        q = self.get_measurement_vector(spikes)
        G = self.get_measurement_operator(spikes)
        G_pl = np.linalg.pinv(G, rcond=cond_n)

        kernels = self._get_kernels_from_spikes(spikes, t)
        x = G_pl.dot(q).dot(kernels)
        return x

    def get_measurement_operator(self, spikes):
        """
        Provides the measurement matrix G required to perform the decoding
        in the decode method of this class
        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the spike times of the multiple integrate-and-fire
            time encoding machines encoding the same signal

        RETURNS
        -------
        np.ndarray
            matrix containing the measurement matrix G
        """

        return np.concatenate(
            [
                np.concatenate(
                    [
                        self._get_measurement_bloc(spikes, ch, ch_j)
                        for ch_j in range(self.n_channels)
                    ],
                    axis=1,
                )
                for ch in range(self.n_channels)
            ],
            axis=0,
        )

    def _get_measurement_bloc(self, spikes, ch_i, ch_j):
        """
        computes a bloc of the measurement matrix G that is computed in
        the get_measurement_operator method
        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the spike times of the multiple integrate-and-fire
            time encoding machines encoding the same signal
        ch_i: int
            index of the measuring channel
        ch_j: int
            index of the channel which generates the kernels whose amplitudes
            we would like to recover

        RETURNS
        -------
        np.ndarray
            matrix containing a block of the measurement matrix G
        """

        (
            sum_k_l,
            sum_k1_l1,
            sum_k1_l,
            sum_k_l1,
            diff_l1_l,
        ) = self._get_integral_start_and_end_points(spikes[ch_i], spikes[ch_j])

        return (
            self._kernel_integral(sum_k1_l1)
            - self._kernel_integral(sum_k_l1)
            - self._kernel_integral(sum_k1_l)
            + self._kernel_integral(sum_k_l)
        ) / (diff_l1_l)

    def _get_integral_start_and_end_points(self, spikes_in_ch_i, spikes_in_ch_j):
        """
        returns starting and and end points of different required integrals
        of the kernel used
        PARAMETERS
        ----------
        spikes_in_ch_i: list
            list of spike times of channel ch_i
        spikes_in_ch_j: list
            list of spike times of channel ch_j

        RETURNS
        -------
        list
            list of np.ndarray 2D matrices containing required start and end
            times of integral of the kernel
        """

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

    def _get_kernels_from_spikes(self, spikes, t):
        """
        obtains the kernels that should be applied to recover the
        input signal x
        x(t) = sum_k (y_k kernel(t-s_k))

        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the spike times of the multiple integrate-and-fire
            time encoding machines encoding the same signal
        t: np.ndarray
            vector holding the times at which one would like the recovered signal
            to be sampled

        RETURNS
        -------
        np.ndarray
            matrix with as many rows as there are measurements from the spike times
            and as many columns as there are times at which the signal is sampled
        """

        def get_kernels_per_channel(spikes, ch, t):
            spikes_in_ch = np.atleast_2d(spikes[ch]).T
            t = np.atleast_2d(t)
            kernels = (
                self._kernel(t - spikes_in_ch[1:, :])
                - self._kernel(t - spikes_in_ch[:-1, :])
            ) / (np.diff(spikes_in_ch, axis=0))
            return kernels

        return np.concatenate(
            [get_kernels_per_channel(spikes, ch, t) for ch in range(self.n_channels)]
        )

    def _kernel(self, t):
        """
        Returns the kernel used for recovery, i.e. the integral of the
        function which applies bandlimitation.
        In the case of a nonperiodic function, this is a sinc function.
        In the case of a periodic function, this is a dirichlet function.

        PARAMETERS
        ----------
        t: float or np.ndarray
            time(s) at which the kernel should be sampled

        RETURNS
        -------
        float or np.ndarray
            kernel sampled at time(s) t
        """

        return (
            Helpers.dirichlet_integral(t, self.period, self.n_components)
            if self.periodic
            else Helpers.sinc_integral(t, self.Omega)
        )

    def _kernel_integral(self, t):
        """
        Returns the second integral of the function which applies bandlimitation.
        In the case of a nonperiodic function, this is a sinc function.
        In the case of a periodic function, this is a dirichlet function.

        PARAMETERS
        ----------
        t: float or np.ndarray
            time(s) at which the second integral should be sampled

        RETURNS
        -------
        float or np.ndarray
            second integral sampled at time(s) t
        """

        return (
            Helpers.dirichlet_second_integral(t, self.period, self.n_components)
            if self.periodic
            else Helpers.sinc_second_integral(t, self.Omega)
        )


class MSignalMChannelDecoder(Decoder):
    """
    Class for a decoder that can use spike times emitted by multiple
    integrate-and-fire time encoding machines to reconstruct multiple
    mixed signals fed into the multiple machine

    ATTRIBUTES
    ----------
    params: TEMParams
        Object which holds the parameters of the time encoding machines
    n_channels: int
        counts the number of channels used to perform the encoding
    periodic: bool
        specifies if the decoder aims to recover periodic signals
    Omega: float
        bandwidth of nonperiodic signal
    period: float
        period of periodic signal
    n_components: int
        number of FS components of periodic signal
    sinc_locs: np.ndarray
        list of locations of sincs that make up an aperiodic signals
    """

    def __init__(
        self,
        tem_parameters,
        periodic=False,
        sinc_locs=None,
        Omega=None,
        period=None,
        n_components=None,
    ):
        """
        PARAMETERS
        ----------
        tem_parameters: TEMParams
            Object which holds the parameters of the time encoding machines
        periodic: bool
            specifies if the decoder aims to recover periodic signals
        Omega: float
            bandwidth of nonperiodic signal
        period: float
            period of periodic signal
        n_components: int
            number of FS components of periodic signal
        sinc_locs: array_like
            list of locations of sincs that make up an aperiodic signals
        """

        super().__init__(tem_parameters, periodic, Omega, period, n_components)
        self.sinc_locs = np.array(sinc_locs)

    def __repr__(self):
        repr = "Multi Signal, Multi Channel Decoder object for "
        if self.periodic:
            repr += (
                "periodic bandlimited signals with period "
                + str(self.period)
                + " and "
                + str(self.n_components)
                + " components, "
            )
        else:
            repr += (
                "sum of sinc signals with bandwidth "
                + str(self.Omega)
                + " and sincs at locations "
                + str(self.sinc_locs)
                + ", "
            )
        repr += "with TEM parameters \n" + str(self.params)
        return repr

    def decode(self, spikes, t=None, return_as_param=False):
        """
        Method that decodes mixed signals from their spike times, by
        recovering coefficients of the functions that generate them

        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the spike times of the multiple integrate-and-fire
            time encoding machines encoding the mixed signals
        t: np.ndarray
            vector holding the times at which one would like the recovered signal
            to be sampled
        return_as_param: bool
           specifies if the input signal should be returned in its parametric form
           (True) or sampled at t (False)

        RETURNS
        -------
        np.ndarray or Signal.SignalCollection
            vector containing input signals sampled at times t or parametric form
            of input signals
        """
        """
        Method that decodes mixed signals from their spike times, by
        recovering coefficients of the functions that generate them

        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the spike times of the multiple integrate-and-fire
            time encoding machines encoding the mixed signals
        t: np.ndarray
            vector holding the times at which one would like the recovered signal
            to be sampled
        return_as_param: bool
           specifies if the input signal should be returned in its parametric form
           (True) or sampled at t (False)

        RETURNS
        -------
        np.ndarray or Signal.SignalCollection
            vector containing input signals sampled at times t or parametric form
            of input signals
        """
        self.__dict__.update(self.params.__dict__)

        if self.periodic:
            x_param = self._decode_periodic(spikes)
        else:
            x_param = self._decode_sum_of_sincs(spikes)
        return x_param if return_as_param else x_param.sample(t)

    def _decode_sum_of_sincs(self, spikes):
        """
        Method that decodes mixed signals from their spike times,
        assuming they come from a sum of sincs at known locations

        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the spike times of the multiple integrate-and-fire
            time encoding machines encoding the mixed signals

        RETURNS
        -------
        Signal.bandlimitedSignals
            object containing the parametric form of the recovered input signals
        """

        q = self.get_measurement_vector(spikes)
        integral_measurement_matrix = self._get_sinc_integral_matrix(spikes)
        flat_fwd_mixing = self._flatten_mixing_matrix(self.mixing_matrix)
        flat_bwd_mixing = self._flatten_mixing_matrix(
            np.linalg.pinv(self.mixing_matrix)
        )
        PCS_sampler = self._get_PCS_sampler(spikes)

        operator_inverse = np.linalg.pinv(
            flat_bwd_mixing.dot(PCS_sampler)
            .dot(integral_measurement_matrix)
            .dot(flat_fwd_mixing)
        )

        x_sinc_amps = (
            operator_inverse.dot(flat_bwd_mixing).dot(PCS_sampler).dot(q)
        ).reshape((self.n_signals, len(self.sinc_locs)))

        return SignalCollection.bandlimitedSignals(
            self.Omega, self.sinc_locs, x_sinc_amps
        )

    def _decode_periodic(self, spikes):
        """
        Method that decodes mixed signals from their spike times,
        assuming they are periodic and come from a sum of complex exponentials
        with known exponents

        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the spike times of the multiple integrate-and-fire
            time encoding machines encoding the mixed signals

        RETURNS
        -------
        Signal.periodicBandlimitedSignals
            object containing the parametric form of the recovered input signals
        """

        measurement_vector = self.get_measurement_vector(spikes)
        measurement_operator = self.get_measurement_operator_periodic(spikes)

        recovered_coefficients = (
            np.linalg.pinv(measurement_operator).dot(measurement_vector)
        ).reshape((self.n_signals, 2 * self.n_components - 1))

        return SignalCollection.periodicBandlimitedSignals(
            self.period, self.n_components, recovered_coefficients
        )

    def _get_PCS_sampler(self, spikes):
        """
        gets a matrix which samples piecewise constant signals with discontinuities
        at the spike times (per channel respectively) using a sinc low pass
        filter

        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the spike times of the multiple integrate-and-fire
            time encoding machines encoding the mixed signals

        RETURNS
        -------
        np.ndarray
           sampler matrix to be applied on different amplitudes/values of the
           piecewise constant signal
        """

        PCSSignal = SignalCollection.piecewiseConstantSignals(
            spikes.get_spikes(),
            values=[
                [0] * (spikes.get_n_spikes_of(ch) - 1) for ch in range(self.n_channels)
            ],
        )
        PCS_sampler_normalizer = np.concatenate(
            [np.diff(spikes[ch]) for ch in range(self.n_channels)]
        )
        return (
            PCSSignal.get_sampler_matrix(self.sinc_locs, self.Omega)
            / PCS_sampler_normalizer
        )

    def _get_sinc_integral_matrix(self, spikes):
        """
        matrix that computes the integrals of sincs at the locations
        self.sinc_locs between any two consecutive spike times of an
        integrate-and-fire time encoding machine channel

        PARAMETERS
        ----------
         spikes: SpikeTimes
             holds the spike times of the multiple integrate-and-fire
             time encoding machines encoding the mixed signals

         RETURNS
         -------
         np.ndarray
            sampler matrix to be applied on different amplitudes/values of the
            sincs at self.sinc_locs
        """

        def get_integral_bloc(ch, integral_index):
            spikes_of_ch = spikes[ch]
            integ_up_limit = spikes_of_ch[integral_index + 1]
            integ_low_limit = spikes_of_ch[integral_index]
            return np.atleast_2d(
                Helpers.sinc_integral(integ_up_limit - self.sinc_locs, self.Omega)
                - Helpers.sinc_integral(integ_low_limit - self.sinc_locs, self.Omega)
            )

        return scipy.linalg.block_diag(
            *[
                np.concatenate(
                    [
                        get_integral_bloc(ch, integral_index)
                        for integral_index in range(spikes.get_n_spikes_of(ch) - 1)
                    ]
                )
                for ch in range(self.n_channels)
            ]
        )

    def _flatten_mixing_matrix(self, mixing_matrix):
        """
        returns a version of the mixing_matrix parameter which operates on
        a flattened version of signal coefficients. i.e. the signal coefficients
        usually have shape num_signals x num_coefficients which are then mixed using
        the mixing_matrix. The flattened mixing matrix instead operates on a vector
        of shape num_signals.num_coefficients x 1.

        PARAMETERS
        ----------
        mixing_matrix: array_like
            matrix to be flattened

        RETURNS
        -------
        np.ndarray
            flattened mixing matrix that now has mixing_matrix.shape[0] x mixing_matrix.shape[1]
            rows and as many columns
        """
        mixing_matrix = np.array(mixing_matrix)

        return np.concatenate(
            [
                np.kron(
                    mixing_matrix[signal_index, :],
                    np.eye(1, len(self.sinc_locs), sinc_index),
                )
                for signal_index in range(mixing_matrix.shape[0])
                for sinc_index in range(len(self.sinc_locs))
            ]
        )

    def get_measurement_operator_periodic(self, spikes):
        """
        Provides the measurement operator that is applied to the complex
        exponential coefficients when the signal is periodic bandlimited

        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the spike times of the multiple integrate-and-fire
            time encoding machines encoding mixed signals

        RETURNS
        -------
        np.ndarray
            matrix containing the measurement operator
        """

        FS_components = (
            1j
            * 2
            * np.pi
            / self.period
            * np.arange(-self.n_components + 1, self.n_components, 1)
        )

        def get_measurement_operator_bloc(ch):
            spikes_in_ch = spikes[ch]
            a_ch = np.atleast_2d(self.mixing_matrix[ch, :])
            integrals = Helpers.exp_int(
                FS_components, spikes_in_ch[:-1:], spikes_in_ch[1::]
            )
            return np.real(
                np.transpose(np.multiply.outer(a_ch, integrals), (3, 1, 2, 0)).reshape(
                    (len(spikes_in_ch) - 1, -1)
                )
            )

        return np.concatenate(
            [get_measurement_operator_bloc(ch) for ch in range(self.n_channels)], axis=0
        )

    def get_integral_start_end_coordinates(self, spikes, TEM_locations):
        """
        returns start ane end coordinates of the integrals that are imposed
        by the spike times in spikes and using the locations in TEM_locations

        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the spike times of the multiple integrate-and-fire
            time encoding machines encoding mixed signals
        TEM_locations: list
            list of self.n_channel tuples, each of which holds and x- and a y-coordinate
            representation the direction in space that the time encoding machine
            is observing

        RETURNS
        -------
        np.ndarray
            Matrix which as many rows as spike time pairs and where each row holds information
            about the time encoding machine's x coordinate, it's y coordinate and the integral
            lower limit (start time)
        np.ndarray
            Matrix which as many rows as spike time pairs and where each row holds information
            about the time encoding machine's x coordinate, it's y coordinate and the integral
            upper limit (end time)
        """
        start_coordinates = np.concatenate(
            [
                np.atleast_2d(
                    [TEM_locations[ch][0], TEM_locations[ch][1], spikes[ch][i]]
                )
                for ch in range(self.n_channels)
                for i in range(len(spikes[ch]) - 1)
            ],
        )
        end_coordinates = np.concatenate(
            [
                np.atleast_2d(
                    [TEM_locations[ch][0], TEM_locations[ch][1], spikes[ch][i + 1]]
                )
                for ch in range(self.n_channels)
                for i in range(len(spikes[ch]) - 1)
            ],
        )
        return start_coordinates, end_coordinates


class UnknownMixingDecoder(Decoder):
    """
    Class for a decoder that can use spike times emitted by multiple
    integrate-and-fire time encoding machines to reconstruct multiple
    mixed low-rank signals fed into the multiple machine

    ATTRIBUTES
    ----------
    params: TEMParams
        Object which holds the parameters of the time encoding machines
    n_channels: int
        counts the number of channels used to perform the encoding
    periodic: bool
        specifies if the decoder aims to recover periodic signals
    Omega: float
        bandwidth of nonperiodic signal
    period: float
        period of periodic signal
    n_components: int
        number of FS components of periodic signal
    sinc_locs: np.ndarray
        list of locations of sincs that make up an aperiodic signals
    """

    def __init__(
        self,
        tem_parameters,
        periodic=False,
        sinc_locs=None,
        Omega=None,
        period=None,
        n_components=None,
    ):
        """
        PARAMETERS
        ----------
        tem_parameters: TEMParams
            Object which holds the parameters of the time encoding machines
        periodic: bool
            specifies if the decoder aims to recover periodic signals
        Omega: float
            bandwidth of nonperiodic signal
        period: float
            period of periodic signal
        n_components: int
            number of FS components of periodic signal
        sinc_locs: array_like
            list of locations of sincs that make up an aperiodic signals
        """

        super().__init__(tem_parameters, periodic, Omega, period, n_components)
        self.sinc_locs = np.array(sinc_locs)

    def __repr__(self):
        repr = "Mixed Multi Signal, Multi Channel Decoder object for "
        if self.periodic:
            repr += (
                "periodic bandlimited signals with unknown mixing, with period "
                + str(self.period)
                + " and "
                + str(self.n_components)
                + " components, "
            )
        else:
            repr += (
                "sum of sinc signals with unknown mixing, with bandwidth "
                + str(self.Omega)
                + " and sincs at locations "
                + str(self.sinc_locs)
                + ", "
            )
        repr += "with TEM parameters \n" + str(self.params)
        return repr

    def get_measurement_matrix(self, spikes):
        """
        Provides the measurement operator that is applied to signal coefficients
        by the spikes of the different time encoding machines

        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the spike times of the multiple integrate-and-fire
            time encoding machines encoding mixed signals

        RETURNS
        -------
        np.ndarray
            matrix containing the measurement operator
        """

        def component_integral(start, end):
            if self.periodic:
                components = np.arange(-self.n_components + 1, self.n_components, 1)
                return Helpers.dirichlet_component_integral(
                    end, self.period, components
                ) - Helpers.dirichlet_component_integral(start, self.period, components)
            else:
                return Helpers.sinc_integral(
                    end - self.sinc_locs, self.Omega
                ) - Helpers.sinc_integral(start - self.sinc_locs, self.Omega)

        measurement_matrices = [
            np.concatenate(
                [
                    np.atleast_2d(
                        component_integral(
                            spikes[ch][sp_i],
                            spikes[ch][sp_i + 1],
                        )
                    )
                    for sp_i in range(spikes.get_n_spikes_of(ch) - 1)
                ],
            )
            for ch in range(self.n_channels)
        ]
        return scipy.linalg.block_diag(*measurement_matrices)

    def decode(self, spikes, rank, t=None, return_as_param=False):
        """
        Method that decodes mixed, low-rank signals from their spike times, by
        recovering coefficients of the functions that generate them while
        enforcing low rank constraints

        PARAMETERS
        ----------
        spikes: SpikeTimes
            holds the spike times of the multiple integrate-and-fire
            time encoding machines encoding the mixed signals
        t: np.ndarray
            vector holding the times at which one would like the recovered signal
            to be sampled
        return_as_param: bool
           specifies if the input signal should be returned in its parametric form
           (True) or sampled at t (False)

        RETURNS
        -------
        np.ndarray or Signal.SignalCollection
            vector containing input signals sampled at times t or parametric form
            of input signals
        """

        shape = (
            self.n_channels,
            len(self.sinc_locs) if not self.periodic else 2 * self.n_components - 1,
        )
        G = self.get_measurement_matrix(spikes)
        q = self.get_measurement_vector(spikes)
        G_inv = np.linalg.pinv(G)
        C_y = Helpers.singular_value_projection_w_matrix(
            shape, G_inv.dot(G), G_inv.dot(q.T), rank, tol=1e-5, lr=0.5
        )

        if not self.periodic:
            y_param = SignalCollection.bandlimitedSignals(
                self.Omega, self.sinc_locs, sinc_amps=C_y
            )
        else:
            y_param = SignalCollection.periodicBandlimitedSignals(
                self.period, self.n_components, C_y
            )
        return y_param if return_as_param else y_param.sample(t)
