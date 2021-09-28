from src import *

class Layer(object):
    def __init__(self, num_inputs, num_outputs, weight_matrix = None):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weight_matrix = np.zeros((self.num_outputs, self.num_inputs))
        self.tem_params = TEMParams(kappa = 1, delta = 1, b = 1, mixing_matrix = self.weight_matrix)
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

        weight_matrix = np.zeros((self.num_outputs, self.num_inputs))

        for n_o in range(self.num_outputs):
            spiking_output = spike_times.get_spikes_of(n_o)
            measurement_results = -self.tem_params.b[n_o]*(spiking_output[1:]-spiking_output[:-1]) + 2 * self.tem_params.kappa[n_o] * self.tem_params.delta[n_o]
            measurement_matrix = np.zeros((len(measurement_results), self.num_inputs))
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
                spiking_output = spike_times[n_e].get_spikes_of(n_o)
                measurement_results_n_e = -self.tem_params.b[n_o]*(spiking_output[1:]-spiking_output[:-1]) + 2 * self.tem_params.kappa[n_o] * self.tem_params.delta[n_o]
                measurement_matrix_n_e = np.zeros((len(measurement_results_n_e), self.num_inputs))
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

