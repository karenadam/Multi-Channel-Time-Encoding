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

    def get_ex_measurement_pairs(
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
                -self.tem_params.b[n_o] * (np.diff(spike_times[n_o]))
                + 2 * self.tem_params.kappa[n_o] * self.tem_params.delta[n_o]
            )
            for n_o in range(self.num_outputs)
        ]

        integrals = [
            Helpers.exp_int(
                exponents,
                s_t[:-1],
                s_t[1:],
            ).T
            for s_t in spike_times
        ]

        measurement_matrices = [
            (np.array(input.coefficient_values).dot(integ.T)).T for integ in integrals
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

    def learn_weight_matrix_from_one_ex(
        self, input: Signal.SignalCollection, spike_times: Spike_Times
    ):
        assert input.n_signals == self.num_inputs
        assert spike_times.n_channels == self.num_outputs

        (
            measurement_matrices,
            measurement_results,
        ) = self.get_ex_measurement_pairs(input, spike_times)

        self.weight_matrix = self.get_weight_matrix_from_parallel_measurements(
            measurement_matrices, measurement_results
        )

    def get_m_ex_measurement_pairs(self, input: list, spike_times: list):
        num_examples = len(input)
        assert num_examples == len(spike_times)

        ex_measurement_pairs = [
            self.get_ex_measurement_pairs(input[n_e], spike_times[n_e])
            for n_e in range(num_examples)
        ]
        measurement_matrices = [ex_m_p[0] for ex_m_p in ex_measurement_pairs]
        measurement_results = [ex_m_p[1] for ex_m_p in ex_measurement_pairs]

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

    def learn_weight_matrix_from_m_ex(self, input: list, spike_times: list):
        assert (input_e.n_signals == self.num_inputs for input_e in input)
        assert (
            spike_times_e.n_channels == self.num_outputs
            for spike_times_e in spike_times
        )
        (
            measurement_matrices,
            measurement_results,
        ) = self.get_m_ex_measurement_pairs(input, spike_times)

        self.weight_matrix = self.get_weight_matrix_from_parallel_measurements(
            measurement_matrices, measurement_results
        )

        return

    def get_dirac_times_from_fsc(self, preactivation_fsc, n_fsc: int, period: float):
        filter_length = n_fsc + 1
        a_filter = src.FRISignal.AnnihilatingFilter(preactivation_fsc, filter_length)
        filter_poly = np.polynomial.polynomial.Polynomial(
            a_filter.get_filter_coefficients()
        )
        roots = filter_poly.roots()
        recovered_times = np.sort(
            np.mod(np.angle(roots) * period / (2 * np.pi), period)
        )
        return recovered_times

    def get_fsc_of_unit_diracs(self, recovered_times, n_fsc, period):
        def get_fsc_of_unit_dirac(time: float, n_fsc, period):
            return src.FRISignal.FRISignal(
                np.array([time]), np.array([1]), period
            ).get_fourier_series(np.arange(-n_fsc + 1, n_fsc, 1).T)

        return np.concatenate(
            [
                np.atleast_2d(get_fsc_of_unit_dirac(t, n_fsc, period))
                for t in recovered_times
            ],
        )

    def learn_spike_input_and_weight_matrix_from_one_example(
        self, spike_times: Spike_Times, n_fsc: int, period: float
    ):
        recovered_times, coefficients = self.get_diracs_from_spikes(
            spike_times, n_fsc, period
        )

        self.weight_matrix = sklearn.cluster.k_means(coefficients.T, self.num_inputs)[0]
        return recovered_times

    def get_diracs_from_spikes(
        self, spike_times: Spike_Times, n_fsc: int, period: float
    ):
        #  n_fsc: number of fourier series coefficients
        preactivation_fsc = self.get_preactivation_fsc(
            spike_times, n_fsc, period, real_f_s=False
        )

        recovered_times = self.get_dirac_times_from_fsc(
            preactivation_fsc, n_fsc, period
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
        n_examples = len(spike_times)
        dirac_params = [
            self.get_diracs_from_spikes(s_t, n_fsc, period) for s_t in spike_times
        ]
        dirac_times = [d_p[0] for d_p in dirac_params]
        dirac_coeffs = np.concatenate([d_p[1] for d_p in dirac_params], axis=0)
        centroids, labels = sklearn.cluster.k_means(dirac_coeffs, self.num_inputs)[0:2]
        self.weight_matrix = copy.deepcopy(centroids.T)

        def get_prvs_layer_spikes(example_index, input_neuron_index):
            example_labels = labels[example_index * n_fsc : (example_index + 1) * n_fsc]
            return dirac_times[example_index][
                np.where(example_labels == input_neuron_index)
            ]

        return [
            [get_prvs_layer_spikes(n_e, n_i) for n_i in range(self.num_inputs)]
            for n_e in range(n_examples)
        ]

    # TODO maybe this can go into/be used directly from decoder code
    def get_preactivation_fsc(
        self,
        spike_times: Spike_Times,
        n_fsc: int,
        period: float,
        real_f_s: bool = True,
    ):
        assert spike_times.n_channels == self.num_outputs
        num_unknowns_to_estimate = 2 * n_fsc - 1 if real_f_s else 2 * (2 * n_fsc - 1)
        num_f_s_symmetry_constraints = n_fsc - 1 if real_f_s else 2 * (n_fsc - 1) + 1
        f_s_coefficients = np.zeros((self.num_outputs, num_unknowns_to_estimate))

        for n_o in range(self.num_outputs):
            spiking_output = spike_times[n_o]

            exponents = 1j * 2 * np.pi / period * np.arange(-n_fsc + 1, n_fsc, 1)

            integrals = Helpers.exp_int(
                exponents, spiking_output[:-1], spiking_output[1:]
            )

            if real_f_s:
                measurement_matrix = np.real(integrals).T
            else:
                measurement_matrix = np.concatenate(
                    [np.real(integrals.T), -np.imag(integrals.T)], axis=1
                )

            def conjugate_symmetry_constraint(n_f_s):
                equal_real_constraint = np.eye(
                    1, num_unknowns_to_estimate, n_f_s
                ) - np.eye(1, num_unknowns_to_estimate, 2 * n_fsc - 2 - n_f_s)
                opposite_imag_constraint = np.eye(
                    1, num_unknowns_to_estimate, -1 - n_f_s
                ) - np.eye(
                    1,
                    num_unknowns_to_estimate,
                    2 * n_fsc - 1 + n_fsc - 1,
                )
                return np.concatenate([equal_real_constraint, opposite_imag_constraint])

            conjugate_symmetry_constraints = np.concatenate(
                [conjugate_symmetry_constraint(n_f_s) for n_f_s in range(n_fsc - 1)]
            )
            measurement_matrix = np.concatenate(
                [
                    measurement_matrix,
                    conjugate_symmetry_constraints,
                    np.eye(
                        1,
                        num_unknowns_to_estimate,
                        2 * n_fsc - 1 + n_fsc - 1,
                    ),
                ]
            )

            measurement_results = np.zeros(measurement_matrix.shape[0])
            measurement_results[: len(spiking_output) - 1] = (
                -self.tem_params.b[n_o] * (spiking_output[1:] - spiking_output[:-1])
                + 2 * self.tem_params.kappa[n_o] * self.tem_params.delta[n_o]
            )

            f_s_coefficients[n_o, :] = (
                np.linalg.lstsq(measurement_matrix, measurement_results, rcond=1e-12)[0]
            ).T

        return (
            f_s_coefficients
            if real_f_s
            else (
                f_s_coefficients[:, : 2 * n_fsc - 1]
                + 1j * f_s_coefficients[:, 2 * n_fsc - 1 :]
            )
        )
