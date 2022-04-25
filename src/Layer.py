from src import *
import sklearn.cluster

class Layer(object):
    def __init__(self, num_inputs, num_outputs, weight_matrix = None, tem_params = None):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weight_matrix = np.zeros((self.num_outputs, self.num_inputs))
        if tem_params is None:
            self.tem_params = TEMParams(kappa = 1, delta = 1, b = 1, mixing_matrix = self.weight_matrix)
        else:
            self.tem_params = tem_params
        if weight_matrix is not None:
            self.set_weight_matrix(weight_matrix)

    def set_weight_matrix(self, weight_matrix):
        assert len(weight_matrix.shape) == 2
        assert weight_matrix.shape[0] == self.num_outputs
        assert weight_matrix.shape[1] == self.num_inputs
        self.weight_matrix = copy.deepcopy(weight_matrix)
        self.tem_params.mixing_matrix = copy.deepcopy(self.weight_matrix)
        
    def learn_weight_matrix_from_one_signal(self, input: Signal.SignalCollection, spike_times: Spike_Times):
        assert input.n_signals == self.num_inputs
        assert spike_times.n_channels == self.num_outputs

        weight_matrix = np.zeros((self.num_outputs, self.num_inputs),  dtype='complex')

        for n_o in range(self.num_outputs):
            spiking_output = spike_times.get_spikes_of(n_o)
            measurement_results = -self.tem_params.b[n_o]*(spiking_output[1:]-spiking_output[:-1]) + 2 * self.tem_params.kappa[n_o] * self.tem_params.delta[n_o]
            measurement_matrix = np.zeros((len(measurement_results), self.num_inputs), dtype='complex')
            # TODO: move this to signal code

            exponents = (
                1j * 2 * np.pi / input.period * np.arange(-input._n_components + 1, input._n_components, 1)
            )

            for n_s in range(len(spiking_output) - 1):
                integrals = Helpers.exp_int(
                    exponents, [spiking_output[n_s]], [spiking_output[n_s + 1]]
                )
                measurement_matrix[n_s, :] = (
                    np.array(input.coefficient_values).dot(integrals)
                ).flatten()

            weight_matrix[n_o, :] = measurement_results.dot((measurement_matrix.T.dot(np.linalg.pinv(measurement_matrix.dot(measurement_matrix.T)))).T)
        self.weight_matrix = copy.deepcopy(weight_matrix)

    def learn_weight_matrix_from_multi_signals(self, input: list, spike_times: list):
        assert (input_e.n_signals == self.num_inputs for input_e in input)
        assert (spike_times_e.n_channels == self.num_outputs for spike_times_e in spike_times)

        num_examples = len(input)
        assert num_examples == len(spike_times)

        weight_matrix = np.zeros((self.num_outputs, self.num_inputs))

        for n_o in range(self.num_outputs):
            measurement_results = np.zeros((0,))
            measurement_matrix = np.zeros((0,self.num_inputs))

            for n_e in range(num_examples):
                spiking_output = spike_times[n_e].get_spikes_of(n_o).flatten()
                measurement_results_n_e = -self.tem_params.b[n_o]*(spiking_output[1:]-spiking_output[:-1]) + 2 * self.tem_params.kappa[n_o] * self.tem_params.delta[n_o]
                measurement_matrix_n_e = np.zeros((len(measurement_results_n_e), self.num_inputs), dtype='complex')
                # TODO: move this to signal code

                exponents = (
                    1j * 2 * np.pi / input[n_e].period * np.arange(-input[n_e]._n_components + 1, input[n_e]._n_components, 1)
                )

                for n_s in range(len(spiking_output) - 1):
                    integrals = Helpers.exp_int(
                        exponents, [spiking_output[n_s]], [spiking_output[n_s + 1]]
                    )
                    measurement_matrix_n_e[n_s, :] = (
                        np.array(input[n_e].coefficient_values).dot(integrals)
                    ).flatten()

                measurement_matrix = np.concatenate((measurement_matrix, measurement_matrix_n_e))
                measurement_results = np.concatenate((measurement_results,measurement_results_n_e))

            weight_matrix[n_o, :] = measurement_results.dot((measurement_matrix.T.dot(np.linalg.pinv(measurement_matrix.dot(measurement_matrix.T)))).T)
        self.weight_matrix = copy.deepcopy(weight_matrix)

    def learn_spike_input_and_weight_matrix_from_one_example(self, spike_times: Spike_Times, num_f_s_coefficients: int, period: float):
        preactivation_f_s_coefficients = self.get_preactivation_f_s_coefficients(spike_times, num_f_s_coefficients, period, real_f_s= False)

        filter_length = num_f_s_coefficients+1
        a_filter = src.FRISignal.AnnihilatingFilter(preactivation_f_s_coefficients, filter_length)
        filter_poly = np.polynomial.polynomial.Polynomial(a_filter.get_filter_coefficients())
        roots = filter_poly.roots()
        recovered_times = np.sort(np.mod(np.angle(roots) * period / (2 * np.pi), period))

        spike_times_corresponding_coefficients = np.zeros((len(recovered_times), 2*num_f_s_coefficients-1), dtype = 'complex')
        for n_t in range(len(recovered_times)):
            fri_signal = src.FRISignal.FRISignal(np.array([recovered_times[n_t]]),  np.array([1]), period)
            spike_times_corresponding_coefficients[n_t, :] = fri_signal.get_fourier_series(np.arange(-num_f_s_coefficients+1, num_f_s_coefficients,1).T)

        coefficients = np.zeros((self.num_outputs, len(recovered_times)))
        for n_o in range(self.num_outputs):
            coefficients[n_o,:] = np.real((np.linalg.lstsq(spike_times_corresponding_coefficients.T, preactivation_f_s_coefficients[n_o,:], rcond = 1e-12)[0]).T)

        centroids, labels = sklearn.cluster.k_means(coefficients.T, self.num_inputs)[0:2]

        self.weight_matrix = copy.deepcopy(centroids.T)

        return recovered_times

    def learn_spike_input_and_weight_matrix_from_multi_example(self, spike_times: list, num_f_s_coefficients: int,
                                                             period: float):
        n_examples = len(spike_times)
        out_nodes_f_s_coeffs = []
        dirac_times = []
        dirac_coeffs = np.zeros((self.num_outputs, 0))
        for n_e in range(n_examples):
            preactivation_f_s_coefficients = self.get_preactivation_f_s_coefficients(spike_times[n_e], num_f_s_coefficients,
                                                                                     period, real_f_s=False)
            out_nodes_f_s_coeffs.append(preactivation_f_s_coefficients)

            filter_length = num_f_s_coefficients + 1
            a_filter = src.FRISignal.AnnihilatingFilter(preactivation_f_s_coefficients, filter_length)
            filter_poly = np.polynomial.polynomial.Polynomial(a_filter.get_filter_coefficients())
            roots = filter_poly.roots()
            recovered_times = np.sort(np.mod(np.angle(roots) * period / (2 * np.pi), period))
            dirac_times.append(recovered_times)

            spike_times_corresponding_coefficients = np.zeros((len(recovered_times), 2 * num_f_s_coefficients - 1),
                                                              dtype='complex')
            for n_t in range(len(recovered_times)):
                fri_signal = src.FRISignal.FRISignal(np.array([recovered_times[n_t]]), np.array([1]), period)
                spike_times_corresponding_coefficients[n_t, :] = fri_signal.get_fourier_series(
                    np.arange(-num_f_s_coefficients + 1, num_f_s_coefficients, 1).T)

            coefficients = np.zeros((self.num_outputs, len(recovered_times)))
            for n_o in range(self.num_outputs):
                coefficients[n_o, :] = np.real((np.linalg.lstsq(spike_times_corresponding_coefficients.T,
                                                                preactivation_f_s_coefficients[n_o, :], rcond=1e-12)[0]).T)
            dirac_coeffs = np.concatenate((dirac_coeffs, coefficients), axis = 1)

        centroids, labels = sklearn.cluster.k_means(dirac_coeffs.T, self.num_inputs)[0:2]

        self.weight_matrix = copy.deepcopy(centroids.T)

        in_nodes_dirac_times = []

        for n_e in range(n_examples):
            example_input_diracs =  []
            for n_i in range(self.num_inputs):
                example_labels = labels[n_e*num_f_s_coefficients:(n_e+1)*num_f_s_coefficients]
                example_input_diracs.append(dirac_times[n_e][np.where(example_labels == n_i)])
            in_nodes_dirac_times.append(example_input_diracs)

        return in_nodes_dirac_times

    # TODO maybe this can go into/be used directly from decoder code
    def get_preactivation_f_s_coefficients(self, spike_times: Spike_Times, num_f_s_coefficients: int, period: float, real_f_s: bool = True):
        assert spike_times.n_channels == self.num_outputs
        if real_f_s:
            f_s_coefficients = np.zeros((self.num_outputs, 2*num_f_s_coefficients -1 ))
        else:
            f_s_coefficients = np.zeros((self.num_outputs, 2*(2*num_f_s_coefficients -1 )))
        weight_matrix = np.zeros((self.num_outputs, self.num_inputs))

        for n_o in range(self.num_outputs):
            spiking_output = spike_times.get_spikes_of(n_o)
            if not real_f_s:
                measurement_results = np.zeros((len(spiking_output)-1 + 2*(num_f_s_coefficients -1 +1)))
                measurement_results[0:len(spiking_output)-1] = -self.tem_params.b[n_o] * (spiking_output[1:] - spiking_output[:-1]) + 2 * \
                                      self.tem_params.kappa[n_o] * self.tem_params.delta[n_o]
                measurement_matrix = np.zeros((len(measurement_results), 2*(2*num_f_s_coefficients-1)))
            else:
                measurement_results = np.zeros((len(spiking_output)-1 + num_f_s_coefficients -1))
                measurement_results[:len(spiking_output)-1] = -self.tem_params.b[n_o] * (spiking_output[1:] - spiking_output[:-1]) + 2 * \
                                      self.tem_params.kappa[n_o] * self.tem_params.delta[n_o]
                measurement_matrix = np.zeros((len(measurement_results), 2*num_f_s_coefficients-1))

            # TODO: move this to signal code
            exponents = (
                1j * 2 * np.pi / period * np.arange(-num_f_s_coefficients + 1, num_f_s_coefficients, 1)
            )

            for n_s in range(len(spiking_output) - 1):
                integrals = Helpers.exp_int(
                    exponents, [spiking_output[n_s]], [spiking_output[n_s + 1]]
                )
                if not real_f_s:
                    measurement_matrix[n_s, :2*num_f_s_coefficients-1] = np.real(integrals.flatten())
                    measurement_matrix[n_s, 2*num_f_s_coefficients-1:] = -np.imag(integrals.flatten())

                if real_f_s:
                    measurement_matrix[n_s,:] = np.real(integrals).flatten()

            if not real_f_s:
                for n_f_s in range(num_f_s_coefficients-1):
                    measurement_matrix[len(spiking_output)-1+2*n_f_s,n_f_s] = 1
                    measurement_matrix[len(spiking_output)-1+2*n_f_s,2*num_f_s_coefficients-2-n_f_s] = -1
                    measurement_matrix[len(spiking_output)-1+2*n_f_s+1,-1-n_f_s] = 1
                    measurement_matrix[len(spiking_output)-1+2*n_f_s+1,2*num_f_s_coefficients-1+n_f_s] = 1
                measurement_matrix[-1, 2*num_f_s_coefficients-1+num_f_s_coefficients-1] = 1

            f_s_coefficients[n_o,:] = (np.linalg.lstsq(measurement_matrix, measurement_results, rcond = 1e-12)[0]).T
        if not real_f_s:
            return f_s_coefficients[:,:2*num_f_s_coefficients-1]+1j*f_s_coefficients[:,2*num_f_s_coefficients-1:]
        else:
            return f_s_coefficients

