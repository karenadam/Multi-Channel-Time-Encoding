from src import *
import sklearn.cluster


class Layer(object):
    """
    Class which implements a layer of a spiking neural network

    Attributes
    ----------
    num_inputs: int
        number of inputs to the layer
    num_outputs: int
        number of outputs of the layer
    weight_matrix: np.ndarray
        2D array of floats, weight matrix that transforms input to layer
        to the input to the nodes
    tem_params: TEMParams
        contains the parameters of the TEMs that are the nodes of the layer
    """

    def __init__(self, num_inputs, num_outputs, weight_matrix=None, tem_params=None):
        """
        Parameters
        ----------
        num_inputs: int
            number of inputs to the layer
        num_outputs: int
            number of outputs of the layer
        weight_matrix: np.ndarray
            2D array of floats, weight matrix that transforms input to layer
            to the input to the nodes
        tem_params: TEMParams
            contains the parameters of the TEMs that are the nodes of the layer
        """

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weight_matrix = np.zeros((self.num_outputs, self.num_inputs))
        self.tem_params = (
            tem_params
            if tem_params
            else TEMParams(kappa=1, delta=1, b=1, mixing_matrix=self.weight_matrix)
        )
        if weight_matrix is not None:
            self.set_weight_matrix(weight_matrix)

    def __repr__(self):
        return (
            "Layer with "
            + str(self.num_inputs)
            + " inputs, "
            + str(self.num_outputs)
            + " outputs and weight matrix:\n"
            + str(self.weight_matrix)
        )

    def set_weight_matrix(self, weight_matrix):
        """
        Parameters
        ----------
        weight_matrix: np.ndarray
            2D array of floats, weight matrix that transforms the input to the layer
            to the input to the nodes

        Raises
        ------
        ValueError
            If the weight matrix does not have the right shape
        """

        print(weight_matrix.shape)

        if len(weight_matrix.shape) != 2:
            raise ValueError("The weight matrix should have two dimensions")
        if weight_matrix.shape[0] != self.num_outputs:
            raise ValueError(
                "The weight matrix should have "
                + str(self.num_outputs)
                + " outputs but has "
                + str(weight_matrix.shape[0])
                + " outputs instead"
            )

        if weight_matrix.shape[1] != self.num_inputs:
            raise ValueError(
                "The weight matrix should have "
                + str(self.num_inputs)
                + " inputs but has "
                + str(weight_matrix.shape[1])
                + " inputs instead"
            )
        self.weight_matrix = copy.deepcopy(weight_matrix)
        self.tem_params.mixing_matrix = copy.deepcopy(self.weight_matrix)

    def get_ex_measurement_pairs(
        self, input: signals.signal_collection, spike_times: SpikeTimes
    ):
        """
        returns the measurement matrix and vector corresponding to the weight matrix of
        each node of the layer for an example containing an input output pair

        Parameters
        ----------
        input: signals.signal_collection
            input to the layer in this example
        spike_times: SpikeTimes
            spike time output of the layer in this example

        Returns
        -------
        list
            list of 2D np.ndarray objects, each of which represents the measurement matrix
            applied to the weights of one of the nodes of the layer
        list
            list of vectors (in the form of 2D np.ndarray objects), each of which represents
            the result of the application of the corresponding measurement matrix to the
            weights of the corresponding node of the layer
        """

        exponents = (
            1j
            * 2
            * np.pi
            / input.period
            * np.arange(-input._n_components + 1, input._n_components, 1)
        )

        measurement_results = [
            np.atleast_2d(
                -self.tem_params.b[n_o] * (np.diff(spike_times[n_o]))
                + 2 * self.tem_params.kappa[n_o] * self.tem_params.delta[n_o]
            ).T
            for n_o in range(self.num_outputs)
        ]

        integrals = [
            helpers.kernels.exp_int(
                exponents,
                s_t[:-1],
                s_t[1:],
            )
            for s_t in spike_times
        ]

        measurement_matrices = [
            (np.array(input.coefficient_values).dot(integ)).T for integ in integrals
        ]

        return measurement_matrices, measurement_results

    def get_weight_matrix_from_parallel_measurements(
        self, measurement_matrices, measurement_results
    ):
        """
        returns the weight matrix that correspond to the measurement_matrices and
        measurement_results applied in parallel to the weight matrices of different
        nodes of the layer

        Parameters
        ----------
        measurement_matrices: list
            list of 2D np.ndarray objects, each of which represents the measurement matrix
            applied to the input of one of the nodes of the layer
        measurement_results: list
            list of vectors (in the form of 2D np.ndarray objects), each of which represents
            the result of the application of the corresponding measurement matrix to the
            corresponding nodes of the layer


        Returns
        -------
        np.ndarray
            The weight matrix of the layer that solves for the given measurement matrices
            and results
        """

        block_diagonal_measurement_matrix = scipy.linalg.block_diag(
            *measurement_matrices
        )
        concatenated_results = np.concatenate(measurement_results)
        return np.linalg.lstsq(block_diagonal_measurement_matrix, concatenated_results)[
            0
        ].reshape((self.num_outputs, self.num_inputs))

    def learn_weight_matrix_from_one_ex(
        self, input: signals.signal_collection, spike_times: spike_times
    ):
        """
        learns the weight matrix that solves (in the least-squares sense) for the
        input/output combination provided in one example

        Parameters
        ----------
        input: signals.signal_collection
            input to the layer in this example
        spike_times: SpikeTimes
            spike time output of the layer in this example

        Raises
        ------
        ValueError
            if the number of inputs or outputs of the example does not match the number of
            inputs or outputs (respectively) of the layer
        """

        (
            measurement_matrices,
            measurement_results,
        ) = self.get_ex_measurement_pairs(input, spike_times)

        self.set_weight_matrix(
            self.get_weight_matrix_from_parallel_measurements(
                measurement_matrices, measurement_results
            )
        )

    def get_m_ex_measurement_pairs(self, input: list, spike_times: list):
        """
        returns the measurement matrix and vector corresponding to the weights for each node of
        the layer for multiple examples

        Parameters
        ----------
        input: list
            list of signals.signal_collection objects, input to the layer in the different examples
        spike_times: list
            list of SpikeTimes objects, spike time output of the layer in the different examples

        Raises
        ------
        ValueError
            if the number of inputs and outputs given does not match

        Returns
        -------
        list
            list of 2D np.ndarray objects, each of which represents the measurement matrix
            applied to the weights of one of the nodes of the layer
        list
            list of vectors (in the form of 2D np.ndarray objects), each of which represents
            the result of the application of the corresponding measurement matrix to the
            weights of the corresponding node of the layer
        """

        num_examples = len(input)
        if len(spike_times) != len(input):
            raise ValueError(
                "The number of inputs and outputs you provide does not match"
            )

        ex_measurement_pairs = [
            self.get_ex_measurement_pairs(input[n_e], spike_times[n_e])
            for n_e in range(num_examples)
        ]

        def group_by_output_neuron(tuple_index: int):
            return [
                np.concatenate(
                    [
                        ex_measurement_pairs[n_e][tuple_index][n_o]
                        for n_e in range(num_examples)
                    ],
                )
                for n_o in range(self.num_outputs)
            ]

        measurement_results = group_by_output_neuron(tuple_index=1)
        measurement_matrices = group_by_output_neuron(tuple_index=0)

        return measurement_matrices, measurement_results

    def learn_weight_matrix_from_m_ex(self, input: list, spike_times: list):
        """
        learns the weight matrix that solves (in the least-squares sense) for the
        input/output combinations provided in multiple examples

        Parameters
        ----------
        input: list
            list of signals.signal_collection objects, input to the layer in the different examples
        spike_times: list
            list of SpikeTimes objects, spike time output of the layer in the different examples

        Raises
        ------
        ValueError
            if any of the number of inputs or outputs of the examples does not match the number of
            inputs or outputs (respectively) of the layer
        """

        (
            measurement_matrices,
            measurement_results,
        ) = self.get_m_ex_measurement_pairs(input, spike_times)

        self.set_weight_matrix(
            self.get_weight_matrix_from_parallel_measurements(
                measurement_matrices, measurement_results
            )
        )

        return

    def get_dirac_times_from_fsc(self, fsc, filter_length: int, period: float):
        """
        gets times of diracs from fourier series coefficients of a signal

        Parameters
        ----------
        fsc: np.ndarray
            vector or matrix containing fourier series coefficients of a (or multiple) signal(s)
        filter_length: int
            length of corresponding annihilating filter (= number of diracs +1)
        period: float
            period of the signal of interest

        Returns
        -------
        np.ndarray
            vector containing the timing of the diracs in the signals with fourier
            series coefficients fsc
        """
        a_filter = src.FRISignal.AnnihilatingFilter(fsc, filter_length)
        filter_poly = np.polynomial.polynomial.Polynomial(
            a_filter.get_filter_coefficients()
        )
        roots = filter_poly.roots()
        recovered_times = np.sort(
            np.mod(np.angle(roots) * period / (2 * np.pi), period)
        )
        return recovered_times

    def get_fsc_of_unit_diracs(self, dirac_times, n_fsc, period):
        """
        gets the fourier series coefficients of diracs with different times

        Parameters
        ----------
        dirac_times: np.ndarray
            vector of times at which each of the diracs occur
        n_fsc: int
            number of complex fourier series coefficients (in practice, we compute 2n_fsc of them)
        period: float
            period of the signals of interest

        Returns
        -------
        np.ndarray
            a 2D matrix, where each row holds the fourier series coefficients for
            a dirac of the corresponding time in dirac_times
        """

        def get_fsc_of_unit_dirac(time: float, n_fsc, period):
            return src.FRISignal.FRISignal(
                np.array([time]), np.array([1]), period
            ).get_fourier_series(np.arange(-n_fsc + 1, n_fsc, 1).T)

        return np.concatenate(
            [
                np.atleast_2d(get_fsc_of_unit_dirac(t, n_fsc, period))
                for t in dirac_times
            ],
        )

    def learn_spike_input_and_weight_matrix_from_one_example(
        self, spike_times: spike_times, n_fsc: int, period: float
    ):
        """
        simultaneously learns (and sets) the weight matrix of a layer and returns
        the input spikes that generate the output provided in the example

        Parameters
        ----------
        spike_times: SpikeTimes
            spike time output of the layer in this example
        n_fsc: int
            fourier series coefficients range from index -n_fsc+1 to n_fsc-1
        period: float
            period of the (apriori unknown) input signal

        Returns
        -------
        np.ndarray
            2D matrix where each row contains the spike times of the corresponding
            node of the layer
        """

        recovered_times, coefficients = self.get_diracs_from_spikes(
            spike_times, n_fsc, period
        )

        self.set_weight_matrix(
            sklearn.cluster.k_means(coefficients, self.num_inputs)[0].T
        )
        return recovered_times

    def get_diracs_from_spikes(
        self, spike_times: spike_times, n_fsc: int, period: float
    ):
        """
        retrieves the input spikes (timing and amplitude) that generate the output
        provided in the example

        Parameters
        ----------
        spike_times: SpikeTimes
            spike time output of the layer in this example
        n_fsc: int
            fourier series coefficients range from index -n_fsc+1 to n_fsc-1
        period: float
            period of the (apriori unknown) input signal

        Returns
        -------
        np.ndarray
            2D matrix where each row contains the spike times of the corresponding
            node of the layer
        np.ndarray
            2D matrix where each row contains the spike amplitudes of the corresponding
            node of the layer
        """

        #  n_fsc: number of fourier series coefficients
        preactivation_fsc = self.get_preactivation_fsc(
            spike_times, n_fsc, period, real_f_s=False
        )

        recovered_times = self.get_dirac_times_from_fsc(
            preactivation_fsc, n_fsc + 1, period
        )
        unmixed_fsc = self.get_fsc_of_unit_diracs(recovered_times, n_fsc, period)

        block_diagonal_measurement_matrix = scipy.linalg.block_diag(
            *[unmixed_fsc.T] * self.num_outputs
        )
        coefficients = np.real(
            np.linalg.lstsq(
                block_diagonal_measurement_matrix,
                preactivation_fsc.flatten(),
            )[0].reshape((self.num_outputs, len(recovered_times)))
        ).T

        return recovered_times, coefficients

    def learn_spike_input_and_weight_matrix_from_multi_example(
        self, spike_times: list, n_fsc: int, period: float
    ):
        """
        simultaneously learns (and sets) the weight matrix of a layer and returns
        the different spiking inputs that generate the outputs provided in multiple examples

        Parameters
        ----------
        spike_times: list
            list of SpikeTimes objects, spike time output of the layer in the different examples
        n_fsc: int
            fourier series coefficients range from index -n_fsc+1 to n_fsc-1
        period: float
            period of the (apriori unknown) input signals

        Returns
        -------
        list
            list of list of vectors where each vectors contains the spike times of the corresponding
            node of the layer and example
        """
        n_examples = len(spike_times)
        dirac_params = [
            self.get_diracs_from_spikes(s_t, n_fsc, period) for s_t in spike_times
        ]
        dirac_times = [d_p[0] for d_p in dirac_params]
        dirac_coeffs = np.concatenate([d_p[1] for d_p in dirac_params], axis=0)
        centroids, labels = sklearn.cluster.k_means(dirac_coeffs, self.num_inputs)[0:2]
        self.set_weight_matrix(copy.deepcopy(centroids.T))

        def get_prvs_layer_spikes(example_index, input_neuron_index):
            example_labels = labels[example_index * n_fsc : (example_index + 1) * n_fsc]
            return dirac_times[example_index][
                np.where(example_labels == input_neuron_index)
            ]

        return [
            [get_prvs_layer_spikes(n_e, n_i) for n_i in range(self.num_inputs)]
            for n_e in range(n_examples)
        ]

    def _get_preactivation_fsc_measurement_constraints(
        self,
        spike_times: spike_times,
        n_o: int,
        n_fsc: int,
        period: float,
        real_fsc: bool = False,
    ):
        """
        get measurement constraints on the fourier series coefficients of the
        signal that is input to nonlinearity n_o of the layer (i.e. the signals
        post mixing) for a particular output spike train

        Parameters
        ----------
        spike_times: SpikeTimes
            spike time output of the layer in this example
        n_o: int
            index of output node for which we would like to compute the input's
            fourier series coefficients
        n_fsc: int
            we compute the fourier series coefficients ranging from -n_fsc+1 to n_fsc-1
        period: float
            period of the (apriori unknown) input signal of interest
        real_fsc: bool
            specifies whether the fourier series coefficients take real values (otherwise
            they are assumed to have complex conjugate symmetry)

        Returns
        -------
        np.ndarray
            2D matrix which represents the measurement matrix applied to the fourier series coefficients
            of the input to the node n_o of the layer
        np.ndarray
            vector which represents the result of the application of the  measurement matrix to the fourier
            series coefficients of the input to the node n_o of the layer
        """

        exponents = 1j * 2 * np.pi / period * np.arange(-n_fsc + 1, n_fsc, 1)
        integrals = helpers.kernels.exp_int(
            exponents, spike_times[n_o][:-1], spike_times[n_o][1:]
        )

        if real_fsc:
            measurement_matrix = np.real(integrals).T
        else:
            measurement_matrix = np.concatenate(
                [np.real(integrals.T), -np.imag(integrals.T)], axis=1
            )

        measurement_results = (
            -self.tem_params.b[n_o] * np.diff(spike_times[n_o])
            + 2 * self.tem_params.kappa[n_o] * self.tem_params.delta[n_o]
        )

        return measurement_matrix, measurement_results

    def _extend_with_conj_symm_cstr(self, meas_matrix, meas_results, n_fsc):
        """
        extends a measurement matrix and result with constraints of conjugate symmetry
        imposed on the measured vector

        Parameters
        ----------
        meas_matrix: np.ndarray
            measurement matrix to be extended with conjugate symmetry measurements
        meas_results: np.ndarray
            measurement vector to be extended with zeros
        n_fsc: int
            fourier series coefficients are computed for values ranging from -n_fsc+1 to n_fsc-1

        Returns
        -------
        np.ndarray
            extended measurement matrix
        np.ndarray
            extended measurement results vector
        """

        conj_symm_cstr = complex_vector_constraints(
            2 * n_fsc - 1
        ).get_conjugate_symmetry_constraints()
        meas_matrix = np.concatenate([meas_matrix, conj_symm_cstr])
        size_zero_padding = meas_matrix.shape[0] - len(meas_results)
        meas_results = np.concatenate([meas_results, np.zeros((size_zero_padding))])
        return meas_matrix, meas_results

    def _get_preactivation_fsc_single_neuron(
        self,
        spike_times: spike_times,
        n_o: int,
        n_fsc: int,
        period: float,
        real_fsc: bool = False,
    ):
        """
        get fourier series coefficients of the signal that is input to nonlinearity
        n_o of the layer (i.e. the signals post mixing) for a particular output spike train

        Parameters
        ----------
        spike_times: SpikeTimes
            spike time output of the layer in this example
        n_o: int
            index of output node for which we would like to compute the input's
            fourier series coefficients
        n_fsc: int
            we compute the fourier series coefficients ranging from -n_fsc+1 to n_fsc-1
        period: float
            period of the (apriori unknown) input signal of interest
        real_fsc: bool
            specifies whether the fourier series coefficients take real values (otherwise
            they are assumed to have complex conjugate symmetry)

        Returns
        -------
        np.ndarray
            vector with fourier series coefficients of input to node n_o for this example
        """

        meas_matrix, meas_results = self._get_preactivation_fsc_measurement_constraints(
            spike_times, n_o, n_fsc, period, real_fsc
        )

        if not real_fsc:
            meas_matrix, meas_results = self._extend_with_conj_symm_cstr(
                meas_matrix, meas_results, n_fsc
            )

        f_s_coefficients = np.linalg.lstsq(meas_matrix, meas_results, rcond=1e-12)[0]

        if not real_fsc:
            f_s_coefficients = (
                f_s_coefficients[: 2 * n_fsc - 1]
                + 1j * f_s_coefficients[2 * n_fsc - 1 :]
            )

        return np.atleast_2d(f_s_coefficients)

    def get_preactivation_fsc(
        self,
        spike_times: spike_times,
        n_fsc: int,
        period: float,
        real_f_s: bool = False,
    ):
        """
        get fourier series coefficients of the signal that is input to the nonlinearities
        of the layer (i.e. the signals post mixing) for a particular output spike train

        Parameters
        ----------
        spike_times: SpikeTimes
            spike time output of the layer in this example
        n_fsc: int
            we compute the fourier series coefficients ranging from -n_fsc+1 to n_fsc-1
        period: float
            period of the (apriori unknown) input signal of interest
        real_fsc: bool
            specifies whether the fourier series coefficients take real values (otherwise
            they are assumed to have complex conjugate symmetry)

        Returns
        -------
        np.ndarray
            matrix with fourier series coefficients of input to each node of the network for
            this example
        """
        assert spike_times.n_channels == self.num_outputs

        f_s_coefficients = np.concatenate(
            [
                self._get_preactivation_fsc_single_neuron(
                    spike_times, n_o, n_fsc, period, real_f_s
                )
                for n_o in range(self.num_outputs)
            ]
        )

        return f_s_coefficients
