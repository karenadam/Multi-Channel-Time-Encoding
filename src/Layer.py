from src import *
import sklearn.cluster


class Layer(object):
    def __init__(self, num_inputs, num_outputs, weight_matrix=None, tem_params=None):
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

    def set_weight_matrix(self, weight_matrix):
        assert len(weight_matrix.shape) == 2
        assert weight_matrix.shape == self.num_outputs, self.num_inputs
        self.weight_matrix = copy.deepcopy(weight_matrix)
        self.tem_params.mixing_matrix = copy.deepcopy(self.weight_matrix)

    def get_measurement_operators_and_results_per_example(
        self, input: Signal.SignalCollection, spike_times: Spike_Times
    ):
        exponents = (
            1j
            * 2
            * np.pi
            / input.period
            * np.arange(-input._n_components + 1, input._n_components, 1)
        )

        measurement_results = [
            (
                -self.tem_params.b[n_o] * (np.diff(spike_times.get_spikes_of(n_o)))
                + 2 * self.tem_params.kappa[n_o] * self.tem_params.delta[n_o]
            )
            for n_o in range(self.num_outputs)
        ]

        integrals = [
            Helpers.exp_int(
                exponents,
                spike_times.get_spikes_of(n_o)[:-1],
                spike_times.get_spikes_of(n_o)[1:],
            ).T
            for n_o in range(self.num_outputs)
        ]

        measurement_matrices = [
            (np.array(input.coefficient_values).dot(integrals[n_o].T)).T
            for n_o in range(self.num_outputs)
        ]

        return measurement_matrices, measurement_results

    def get_weight_matrix_from_parallel_measurements(
        self, measurement_matrices, measurement_results
    ):
        block_diagonal_measurement_matrix = scipy.linalg.block_diag(
            *measurement_matrices
        )
        concatenated_results = np.concatenate(measurement_results)
        return np.linalg.lstsq(block_diagonal_measurement_matrix, concatenated_results)[
            0
        ].reshape((self.num_outputs, self.num_inputs))

    def learn_weight_matrix_from_one_signal(
        self, input: Signal.SignalCollection, spike_times: Spike_Times
    ):
        assert input.n_signals == self.num_inputs
        assert spike_times.n_channels == self.num_outputs

        (
            measurement_matrices,
            measurement_results,
        ) = self.get_measurement_operators_and_results_per_example(input, spike_times)

        self.weight_matrix = self.get_weight_matrix_from_parallel_measurements(
            measurement_matrices, measurement_results
        )

    def get_measurement_operators_and_results_multi_examples(
        self, input: list, spike_times: list
    ):
        num_examples = len(input)
        assert num_examples == len(spike_times)

        measurement_matrices = []
        measurement_results = []
        for n_e in range(num_examples):
            (
                temp_meas_matrix,
                temp_meas_results,
            ) = self.get_measurement_operators_and_results_per_example(
                input[n_e], spike_times[n_e]
            )
            measurement_matrices.append(temp_meas_matrix)
            measurement_results.append(temp_meas_results)

        measurement_results = [
            np.concatenate(
                [measurement_results[n_e][n_o] for n_e in range(num_examples)]
            )
            for n_o in range(self.num_outputs)
        ]
        measurement_matrices = [
            np.concatenate(
                [measurement_matrices[n_e][n_o] for n_e in range(num_examples)]
            )
            for n_o in range(self.num_outputs)
        ]
        return measurement_matrices, measurement_results

    def learn_weight_matrix_from_multi_signals(self, input: list, spike_times: list):
        assert (input_e.n_signals == self.num_inputs for input_e in input)
        assert (
            spike_times_e.n_channels == self.num_outputs
            for spike_times_e in spike_times
        )
        (
            measurement_matrices,
            measurement_results,
        ) = self.get_measurement_operators_and_results_multi_examples(
            input, spike_times
        )

        self.weight_matrix = self.get_weight_matrix_from_parallel_measurements(
            measurement_matrices, measurement_results
        )

        return

    def get_input_spike_times_from_F_S_coefficients(
        self, preactivation_f_s_coefficients, num_f_s_coefficients: int, period: float
    ):

        filter_length = num_f_s_coefficients + 1
        a_filter = src.FRISignal.AnnihilatingFilter(
            preactivation_f_s_coefficients, filter_length
        )
        filter_poly = np.polynomial.polynomial.Polynomial(
            a_filter.get_filter_coefficients()
        )
        roots = filter_poly.roots()
        recovered_times = np.sort(
            np.mod(np.angle(roots) * period / (2 * np.pi), period)
        )
        return recovered_times


    def get_unmixed_f_s_coefficients_of_diracs_at(self, recovered_times, num_f_s_coefficients, period):
        unmixed_f_s_coefficients = np.zeros(
            (len(recovered_times), 2 * num_f_s_coefficients - 1), dtype="complex"
        )
        for n_t in range(len(recovered_times)):
            fri_signal = src.FRISignal.FRISignal(
                np.array([recovered_times[n_t]]), np.array([1]), period
            )
            unmixed_f_s_coefficients[n_t, :] = fri_signal.get_fourier_series(
                np.arange(-num_f_s_coefficients + 1, num_f_s_coefficients, 1).T
            )
        return unmixed_f_s_coefficients

    def learn_spike_input_and_weight_matrix_from_one_example(
        self, spike_times: Spike_Times, num_f_s_coefficients: int, period: float
    ):

        preactivation_f_s_coefficients = self.get_preactivation_f_s_coefficients(
            spike_times, num_f_s_coefficients, period, real_f_s=False
        )

        recovered_times = self.get_input_spike_times_from_F_S_coefficients(
            preactivation_f_s_coefficients, num_f_s_coefficients, period
        )
        unmixed_f_s_coefficients = self.get_unmixed_f_s_coefficients_of_diracs_at(recovered_times, num_f_s_coefficients, period)

        block_diagonal_measurement_matrix = scipy.linalg.block_diag(
            *[unmixed_f_s_coefficients.T] * self.num_outputs
        )
        coefficients = np.real(
            np.linalg.lstsq(
                block_diagonal_measurement_matrix,
                preactivation_f_s_coefficients.flatten(),
            )[0].reshape((self.num_outputs, len(recovered_times)))
        ).T

        self.weight_matrix = sklearn.cluster.k_means(coefficients.T, self.num_inputs)[0]
        return recovered_times


    def learn_spike_input_and_weight_matrix_from_multi_example(
        self, spike_times: list, num_f_s_coefficients: int, period: float
    ):
        n_examples = len(spike_times)
        dirac_times = []
        dirac_coeffs = np.zeros((self.num_outputs, 0))
        for n_e in range(n_examples):
            preactivation_f_s_coefficients = self.get_preactivation_f_s_coefficients(
                spike_times[n_e], num_f_s_coefficients, period, real_f_s=False
            )

            recovered_times = self.get_input_spike_times_from_F_S_coefficients(
                preactivation_f_s_coefficients, num_f_s_coefficients, period
            )
            dirac_times.append(recovered_times)

            unmixed_f_s_coefficients = self.get_unmixed_f_s_coefficients_of_diracs_at(recovered_times, num_f_s_coefficients, period)

            block_diagonal_measurement_matrix = scipy.linalg.block_diag(
                *[unmixed_f_s_coefficients.T] * self.num_outputs
            )
            coefficients = np.real(
                np.linalg.lstsq(
                    block_diagonal_measurement_matrix,
                    preactivation_f_s_coefficients.flatten(),
                )[0]).reshape((self.num_outputs, len(recovered_times)))

            dirac_coeffs = np.concatenate((dirac_coeffs, coefficients), axis=1)

        centroids, labels = sklearn.cluster.k_means(dirac_coeffs.T, self.num_inputs)[
            0:2
        ]

        self.weight_matrix = copy.deepcopy(centroids.T)

        in_nodes_dirac_times = []

        for n_e in range(n_examples):
            example_input_diracs = []
            for n_i in range(self.num_inputs):
                example_labels = labels[
                    n_e * num_f_s_coefficients : (n_e + 1) * num_f_s_coefficients
                ]
                example_input_diracs.append(
                    dirac_times[n_e][np.where(example_labels == n_i)]
                )
            in_nodes_dirac_times.append(example_input_diracs)

        return in_nodes_dirac_times

    # TODO maybe this can go into/be used directly from decoder code
    def get_preactivation_f_s_coefficients(
        self,
        spike_times: Spike_Times,
        num_f_s_coefficients: int,
        period: float,
        real_f_s: bool = True,
    ):
        assert spike_times.n_channels == self.num_outputs
        num_unknowns_to_estimate = 2*num_f_s_coefficients-1 if real_f_s else 2*(2 * num_f_s_coefficients - 1)
        num_f_s_symmetry_constraints = num_f_s_coefficients - 1 if real_f_s else 2 * (num_f_s_coefficients - 1)+1
        f_s_coefficients = np.zeros((self.num_outputs, num_unknowns_to_estimate))


        for n_o in range(self.num_outputs):
            spiking_output = spike_times.get_spikes_of(n_o)

            exponents = (
                1j
                * 2
                * np.pi
                / period
                * np.arange(-num_f_s_coefficients + 1, num_f_s_coefficients, 1)
            )

            integrals = Helpers.exp_int(
                exponents, spiking_output[:-1], spiking_output[1:]
            )

            if real_f_s:
                measurement_matrix = np.real(integrals).T
            else:
                measurement_matrix = np.concatenate([np.real(
                    integrals.T
                ), -np.imag(integrals.T)], axis = 1)


            def conjugate_symmetry_constraint(n_f_s):
                equal_real_constraint =  np.eye(1, num_unknowns_to_estimate, n_f_s) - np.eye(1, num_unknowns_to_estimate, 2 * num_f_s_coefficients - 2 - n_f_s)
                opposite_imag_constraint = np.eye(1, num_unknowns_to_estimate, -1-n_f_s) - np.eye(1, num_unknowns_to_estimate,  2 * num_f_s_coefficients - 1 + num_f_s_coefficients - 1)
                return np.concatenate([equal_real_constraint, opposite_imag_constraint])

            conjugate_symmetry_constraints = np.concatenate([conjugate_symmetry_constraint(n_f_s) for n_f_s in range(num_f_s_coefficients-1)])
            measurement_matrix = np.concatenate([measurement_matrix, conjugate_symmetry_constraints, np.eye(1, num_unknowns_to_estimate, 2 * num_f_s_coefficients - 1 + num_f_s_coefficients - 1)])


            measurement_results = np.zeros(measurement_matrix.shape[0]
            )
            measurement_results[: len(spiking_output) - 1] = (
                    -self.tem_params.b[n_o] * (spiking_output[1:] - spiking_output[:-1])
                    + 2 * self.tem_params.kappa[n_o] * self.tem_params.delta[n_o]
                )

            f_s_coefficients[n_o, :] = (
                np.linalg.lstsq(measurement_matrix, measurement_results, rcond=1e-12)[0]
            ).T

        return f_s_coefficients if real_f_s else (f_s_coefficients[:, : 2 * num_f_s_coefficients - 1]
                + 1j * f_s_coefficients[:, 2 * num_f_s_coefficients - 1 :])
