from src import *


class _MultiDimSignal(object):
    def __init__(self, opt):
        self.numDimensions = opt.numDimensions


class MultiDimPeriodicSignal(_MultiDimSignal):
    # Note that freq_domain_samples are assumed to be FFT samples (so the complex conjugate counterpart is not explicitly included in the variable)
    def __init__(self, opt):
        if "time_domain_samples" in opt:
            self.time_domain_samples = opt["time_domain_samples"]
            self.fft = np.fft.fftn(
                self.time_domain_samples
            )  # , s = [int(s/2)*2+1 for s in self.time_domain_samples.shape])
        elif "freq_domain_samples" in opt:
            self.fft = opt["freq_domain_samples"]
            self.time_domain_samples = np.fft.ifftn(self.fft)
        elif "fs_components" in opt:
            raise NotImplementedError
        else:
            raise NotImplementedError
        self.numDimensions = len(self.time_domain_samples.shape)
        self.periods = self.time_domain_samples.shape
        self.n_t_components = int(self.periods[-1] / 2) + 1
        self.dim_frequencies = []
        for dim in range(self.numDimensions):
            self.dim_frequencies.append(self.__get_dim_frequencies(dim))
        self.num_components = [
            len(self.dim_frequencies[nD]) for nD in range(self.numDimensions)
        ]
        self.freq_domain_samples = self.__adjust_freq_domain_samples(self.fft)
        self.__set_fs_components()

    def __get_dim_frequencies(self, dim):
        dim_frequencies = 1 / self.periods[dim] * np.arange(0, self.periods[dim], 1)
        dim_frequencies = 2 * np.pi * (np.remainder(dim_frequencies + 0.5, 1) - 0.5)
        if self.periods[dim] % 2 == 0:
            dim_frequencies = np.insert(
                dim_frequencies, int(self.periods[dim] / 2), np.pi
            )

        return dim_frequencies

    def __adjust_freq_domain_samples(self, freq_samples):
        adjusted_freq_samples = copy.deepcopy(freq_samples)
        for nD in range(self.numDimensions):
            ind = [slice(0, None)] * self.numDimensions
            ind[nD] = int(self.periods[nD] / 2)

            if freq_samples.shape[nD] % 2 == 0:

                adjusted_freq_samples[tuple(ind)] /= 2
                copied_freq_samples = copy.deepcopy(
                    adjusted_freq_samples[tuple(ind)]
                ).conj()
                copied_freq_samples[1:, :] = copied_freq_samples[-1:0:-1, :]
                copied_freq_samples[:, 1:] = copied_freq_samples[:, -1:0:-1]
                adjusted_freq_samples = np.insert(
                    adjusted_freq_samples,
                    int(self.periods[nD] / 2),
                    copied_freq_samples,
                    axis=nD,
                )

        return adjusted_freq_samples

    def __set_fs_components(self):
        self.fs_components = np.roll(self.freq_domain_samples, self.n_t_components, -1)

    def _get_integral_frequency_domain_factors(self, coordinates):
        factors_shape = [1] * (self.numDimensions - 1)
        factors_shape.append(self.num_components[-1])
        factors = np.zeros(factors_shape, dtype=complex)
        factors[:, :, 1:] = 1 / (1j * self.dim_frequencies[-1][1:])
        factors[:, :, 0] = coordinates[-1]

        # print(factors)

        for nD in range(self.numDimensions - 1):
            factors = np.repeat(factors, self.num_components[nD], nD)
        # print("FACTORS", factors)

        exponentials = self._get_exponentials_vector(coordinates)
        return np.multiply(exponentials, factors)

    def _get_exponential_nD_factor(self, coordinates, nD):
        component_nD = coordinates[nD] * self.dim_frequencies[nD]
        component_direction_vector = np.ones((self.numDimensions)).astype(int)
        component_direction_vector[nD] = int(self.num_components[nD])
        return np.reshape(component_nD, component_direction_vector)

    def _get_exponentials_vector(self, coordinates):
        exponents = np.zeros_like(self.freq_domain_samples, dtype="float")
        for nD in range(self.numDimensions):
            exponents += self._get_exponential_nD_factor(coordinates, nD)
        return np.exp(1j * exponents)

    def sample(self, coordinates):
        exponentials = self._get_exponentials_vector(coordinates)
        weighted_exponentials = np.multiply(self.freq_domain_samples, exponentials)
        return np.real(1 / (np.product(self.periods)) * np.sum(weighted_exponentials))

    def get_precise_integral(self, coordinates_end, coordinates_start=None):
        if coordinates_start is None:
            coordinates_start = copy.deepcopy(coordinates_end)
            coordinates_start[-1] = 0
        integral_weights_end = self._get_integral_frequency_domain_factors(
            coordinates_end
        )
        integral_weights_start = self._get_integral_frequency_domain_factors(
            coordinates_start
        )
        integral_op = integral_weights_end - integral_weights_start
        return np.real(
            1
            / (np.product(self.periods))
            * np.sum(np.multiply(self.freq_domain_samples, integral_op))
        )

    def get_time_signal(self, space_coordinates):
        assert len(space_coordinates) == self.numDimensions - 1
        coordinates_t_0 = copy.deepcopy(space_coordinates)
        coordinates_t_0.append(0)
        exponentials = self._get_exponentials_vector(coordinates_t_0)
        weighted_exponentials = np.multiply(
            self.freq_domain_samples, exponentials
        ) / np.product(self.periods)
        time_components = np.sum(
            weighted_exponentials,
            axis=tuple([i for i in range(self.numDimensions - 1)]),
        )
        time_components_reshuffled = np.roll(time_components, int(self.periods[-1] / 2))
        return Signal.periodicBandlimitedSignal(
            self.periods[-1], self.n_t_components, time_components_reshuffled
        )

    def get_coefficients_from_integrals(
        self, integral_start_coordinates, integral_end_coordinates, integrals
    ):
        assert len(integral_start_coordinates) == len(integral_end_coordinates) and len(
            integrals
        ) == len(integral_start_coordinates)

        num_samples = len(integrals)
        linear_operator = np.zeros(
            (
                num_samples + 5 * np.product(self.num_components),
                2 * np.product(self.num_components),
            )
        )
        linear_operator = np.zeros((0, 2 * np.product(self.num_components)))

        augmented_integrals = np.zeros((0, 1))

        for sample_i in range(num_samples):
            integral_start_weights = self._get_integral_frequency_domain_factors(
                integral_start_coordinates[sample_i, :]
            ).flatten()
            integral_end_weights = self._get_integral_frequency_domain_factors(
                integral_end_coordinates[sample_i, :]
            ).flatten()
            linear_operator_i = (
                1
                / np.product(self.periods)
                * (integral_end_weights - integral_start_weights)
            )
            linear_operator = np.append(
                linear_operator,
                np.atleast_2d(
                    np.concatenate(
                        (np.imag(linear_operator_i), np.real(linear_operator_i))
                    )
                ),
                axis=0,
            )

            augmented_integrals = np.append(augmented_integrals, integrals[sample_i])

        n = 0

        def impose_complex_conjugates(indices_1, indices_2):
            op_i = np.zeros(self.num_components)
            op_i[indices_1] += 1
            op_i[indices_2] += 1
            nonlocal linear_operator
            nonlocal n
            linear_operator = np.append(
                linear_operator, np.atleast_2d([0] * linear_operator.shape[1]), 0
            )
            linear_operator[
                num_samples + n, : np.product(self.num_components)
            ] = op_i.flatten()

            n += 1
            op_i = np.zeros(self.num_components)
            op_i[indices_1] += 1
            op_i[indices_2] += -1
            linear_operator = np.append(
                linear_operator, np.atleast_2d([0] * linear_operator.shape[1]), 0
            )
            linear_operator[
                num_samples + n, np.product(self.num_components) :
            ] = op_i.flatten()
            n += 1

        def impose_equal(indices_1, indices_2):
            op_i = np.zeros(self.num_components)
            op_i[indices_1] += 1
            op_i[indices_2] -= 1
            nonlocal linear_operator
            nonlocal n
            linear_operator = np.append(
                linear_operator, np.atleast_2d([0] * linear_operator.shape[1]), 0
            )
            linear_operator = np.append(
                linear_operator, np.atleast_2d([0] * linear_operator.shape[1]), 0
            )
            linear_operator[
                num_samples + n, np.product(self.num_components) :
            ] = op_i.flatten()
            n += 1
            linear_operator[
                num_samples + n, : np.product(self.num_components)
            ] = op_i.flatten()
            n += 1

        def impose_real(indices):
            op_i = np.zeros(self.num_components)
            op_i[indices] = 1
            nonlocal linear_operator
            nonlocal n
            linear_operator = np.append(
                linear_operator, np.atleast_2d([0] * linear_operator.shape[1]), 0
            )
            linear_operator[
                num_samples + n, : np.product(self.num_components)
            ] = op_i.flatten()
            n += 1

        for dim_i in range(self.num_components[0]):
            for dim_j in range(self.num_components[1]):
                for dim_k in range(self.num_components[2]):
                    impose_complex_conjugates(
                        (dim_i, dim_j, dim_k), (-dim_i, -dim_j, -dim_k)
                    )

                    if self.periods[0] % 2 == 0 and dim_i == int(
                        self.num_components[0] / 2
                    ):
                        impose_equal((dim_i, dim_j, dim_k), (-dim_i, dim_j, dim_k))

                    if self.periods[1] % 2 == 0 and dim_j == int(
                        self.num_components[1] / 2
                    ):
                        impose_equal((dim_i, dim_j, dim_k), (dim_i, -dim_j, dim_k))

                    if self.periods[2] % 2 == 0 and dim_k == int(
                        self.num_components[2] / 2
                    ):
                        impose_equal((dim_i, dim_j, dim_k), (dim_i, dim_j, -dim_k))

                    if dim_i == 0 or (
                        self.periods[0] % 2 == 0
                        and (
                            dim_i == int(self.num_components[0] / 2)
                            or dim_i == int(self.num_components[0] / 2) + 1
                        )
                    ):
                        if dim_j == 0 or (
                            self.periods[1] % 2 == 0
                            and (
                                dim_j == int(self.num_components[1] / 2)
                                or dim_j == int(self.num_components[1] / 2) + 1
                            )
                        ):
                            if dim_k > 0:
                                impose_complex_conjugates(
                                    (dim_i, dim_j, dim_k), (dim_i, dim_j, -dim_k)
                                )

                    if dim_i == 0 or (
                        self.periods[0] % 2 == 0
                        and (
                            dim_i == int(self.num_components[0] / 2)
                            or dim_i == int(self.num_components[0] / 2) + 1
                        )
                    ):
                        if dim_k == 0 or (
                            self.periods[2] % 2 == 0
                            and (
                                dim_k == int(self.num_components[2] / 2)
                                or dim_k == int(self.num_components[2] / 2) + 1
                            )
                        ):
                            if dim_j > 0:
                                impose_complex_conjugates(
                                    (dim_i, dim_j, dim_k), (dim_i, -dim_j, dim_k)
                                )

                    if dim_j == 0 or (
                        self.periods[1] % 2 == 0
                        and (
                            dim_j == int(self.num_components[1] / 2)
                            or dim_j == int(self.num_components[1] / 2) + 1
                        )
                    ):
                        if dim_k == 0 or (
                            self.periods[2] % 2 == 0
                            and (
                                dim_k == int(self.num_components[2] / 2)
                                or dim_k == int(self.num_components[2] / 2) + 1
                            )
                        ):
                            if dim_i > 0:
                                impose_complex_conjugates(
                                    (dim_i, dim_j, dim_k), (-dim_i, dim_j, dim_k)
                                )

                    if dim_i == 0 or (
                        self.periods[0] % 2 == 0
                        and (
                            dim_i == int(self.num_components[0] / 2)
                            or dim_i == int(self.num_components[0] / 2) + 1
                        )
                    ):
                        if dim_j == 0 or (
                            self.periods[1] % 2 == 0
                            and (
                                dim_j == int(self.num_components[1] / 2)
                                or dim_j == int(self.num_components[1] / 2) + 1
                            )
                        ):
                            if dim_k == 0 or (
                                self.periods[2] % 2 == 0
                                and (
                                    dim_k == int(self.num_components[2] / 2)
                                    or dim_k == int(self.num_components[2] / 2) + 1
                                )
                            ):
                                impose_real((dim_i, dim_j, dim_k))

        augmented_integrals = np.append(augmented_integrals, [0] * n)
        coefficients_imag_real = np.linalg.lstsq(
            linear_operator, augmented_integrals, rcond=1e-12
        )[0]
        coefficients = (
            coefficients_imag_real[np.product(self.num_components) :]
            - 1j * coefficients_imag_real[: np.product(self.num_components)]
        )
        return coefficients
