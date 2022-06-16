from src import*

class Video(MultiDimPeriodicSignal):

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


    def _get_integral_frequency_domain_factors(self, coordinates):
        factors_shape = [1] * (self.numDimensions - 1)
        factors_shape.append(self.num_components[-1])
        factors = np.zeros(factors_shape, dtype=complex)
        factors[:, :, 1:] = 1 / (1j * self.dim_frequencies[-1][1:])
        factors[:, :, 0] = coordinates[-1]

        for nD in range(self.numDimensions - 1):
            factors = np.repeat(factors, self.num_components[nD], nD)

        exponentials = self._get_exponentials_vector(coordinates)
        return np.multiply(exponentials, factors)

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


    def get_coefficients_from_integrals(
        self, integral_start_coordinates, integral_end_coordinates, integrals
    ):
        num_samples = len(integrals)

        def get_linear_operator(sample_i):
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
            return np.atleast_2d(
                np.concatenate([np.imag(linear_operator_i), np.real(linear_operator_i)])
            )

        cstr = complex_tensor_constraints(self.num_components)
        linear_operator = np.concatenate(
            [
                np.concatenate(
                    [get_linear_operator(sample_i) for sample_i in range(num_samples)]
                ),
                cstr._get_equality_constraints(self.periods),
                cstr.real_center_coefficients_along_single_components(),
                cstr.complex_conjugate_symmetry_along_single_components(),
                cstr.center_point_reflection_complex_conjugates(),
            ]
        )

        augmented_integrals = np.zeros((linear_operator.shape[0]), dtype="complex")
        augmented_integrals[: len(integrals)] = integrals
        coefficients_imag_real = np.linalg.lstsq(
            linear_operator, augmented_integrals, rcond=1e-12
        )[0]
        coefficients = (
            coefficients_imag_real[np.product(self.num_components) :]
            - 1j * coefficients_imag_real[: np.product(self.num_components)]
        )
        return coefficients