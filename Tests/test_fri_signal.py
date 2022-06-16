import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/..")

import src

class TestFriSignal:
    def test_proper_creation(self):
        fri_signal = src.FRISignal.FRISignal(
            np.array([0, 1, 1.5]), np.array([1, 1, 1]), 2.0
        )
        assert np.allclose(fri_signal._dirac_locations, np.array([0, 1, 1.5]))
        assert np.allclose(fri_signal._dirac_weights, np.array([1, 1, 1]))

    def test_proper_FS_coeffs_1(self):
        t_k = np.array([1])
        c_k = np.array([1])
        fri_signal = src.FRISignal.FRISignal(t_k, c_k, 2.0)
        m = np.array([[1, 2, 5]]).T
        assert np.allclose(
            fri_signal.get_fourier_series(m),
            1 / 2 * np.exp(-1j * 2 * np.pi * m / 2.0).T,
        )

    def test_proper_FS_coeffs_2(self):
        t_k = np.array([1, 1.5])
        c_k = np.array([1, -1])
        fri_signal = src.FRISignal.FRISignal(t_k, c_k, 2.0)
        m = np.array([[1, 2, 3, 5]]).T
        assert np.allclose(
            fri_signal.get_fourier_series(m),
            1
            / 2
            * (
                np.exp(-1j * 2 * np.pi * m / 2.0)
                - np.exp(-1j * 2 * 1.5 * np.pi * m / 2.0)
            ).T,
        )

    def test_can_find_annihilator_1_signal(self):
        t_k = np.array([1, 1.5])
        c_k = np.array([1, -1])
        fri_signal = src.FRISignal.FRISignal(t_k, c_k, 2.0)
        M = 3
        f_s_coefficients = fri_signal.get_fourier_series(np.arange(-M, M + 1, 1).T)
        a_filter = src.FRISignal.AnnihilatingFilter(f_s_coefficients)
        result = np.convolve(a_filter.get_filter_coefficients(), f_s_coefficients)
        assert np.allclose(
            result[M - 1 : -M + 1 : 1], np.zeros_like(result[M - 1 : -M + 1 : 1])
        )

    def test_can_find_annihilator_2_signals(self):
        t_k = np.array([1, 1.5])
        c_k_1 = np.array([1, -1])
        c_k_2 = np.array([0.5, 3])
        fri_signal_1 = src.FRISignal.FRISignal(t_k, c_k_1, 2.0)
        fri_signal_2 = src.FRISignal.FRISignal(t_k, c_k_2, 2.0)
        M = 3
        f_s_coefficients = np.zeros((2, 2 * M + 1), dtype="complex")
        f_s_coefficients[0, :] = fri_signal_1.get_fourier_series(
            np.arange(-M, M + 1, 1).T
        )
        f_s_coefficients[1, :] = fri_signal_2.get_fourier_series(
            np.arange(-M, M + 1, 1).T
        )
        a_filter = src.FRISignal.AnnihilatingFilter(f_s_coefficients)
        result_1 = np.convolve(
            a_filter.get_filter_coefficients(), f_s_coefficients[0, :]
        )
        result_2 = np.convolve(
            a_filter.get_filter_coefficients(), f_s_coefficients[1, :]
        )
        assert np.allclose(result_1[M:-M], np.zeros_like(result_1[M:-M]))
        assert np.allclose(result_2[M:-M], np.zeros_like(result_2[M:-M]))

    def test_make_signal_annihilatable(self):
        t_k_1 = np.array([1, 1.5])
        t_k_2 = np.array([0, 1.2])
        c_k_1 = np.array([1, -1])
        c_k_2 = np.array([0.5, 3])
        fri_signal_1 = src.FRISignal.FRISignal(t_k_1, c_k_1, 2.0)
        fri_signal_2 = src.FRISignal.FRISignal(t_k_2, c_k_2, 2.0)
        M = 3
        f_s_coefficients = np.zeros((2, 2 * M + 1), dtype="complex")
        f_s_coefficients[0, :] = fri_signal_1.get_fourier_series(
            np.arange(-M, M + 1, 1).T
        )
        f_s_coefficients[1, :] = fri_signal_2.get_fourier_series(
            np.arange(-M, M + 1, 1).T
        )
        a_filter = src.FRISignal.AnnihilatingFilter(f_s_coefficients[0, :])
        annihilatable_f_s = a_filter.make_annihilatable(fri_signal_2)
        result_1 = np.convolve(
            a_filter.get_filter_coefficients(), f_s_coefficients[0, :]
        )
        result_2 = np.convolve(
            a_filter.get_filter_coefficients(), f_s_coefficients[1, :]
        )
        result_2_updated = np.convolve(
            a_filter.get_filter_coefficients(), annihilatable_f_s
        )
        assert np.allclose(result_1[M:-M], np.zeros_like(result_1[M:-M]))
        assert not np.allclose(result_2[M:-M], np.zeros_like(result_2[M:-M]))
        assert np.allclose(
            result_2_updated[M:-M], np.zeros_like(result_2_updated[M:-M])
        )

    def test_random_sensing_of_f_s_coefficients(self):
        t_k_1 = np.array([1, 1.5])
        c_k_1 = np.array([1, -1])
        c_k_2 = np.array([0.5, 3])
        fri_signal_1 = src.FRISignal.FRISignal(t_k_1, c_k_1, 2.0)
        fri_signal_2 = src.FRISignal.FRISignal(t_k_1, c_k_2, 2.0)
        M = 3
        f_s_coefficients = np.zeros((2, 2 * M + 1), dtype="complex")
        f_s_coefficients[0, :] = fri_signal_1.get_fourier_series(
            np.arange(-M, M + 1, 1).T
        )
        f_s_coefficients[1, :] = fri_signal_2.get_fourier_series(
            np.arange(-M, M + 1, 1).T
        )
        f_s_coefficients_flattened = f_s_coefficients.flatten()
        random_sensing_matrix = np.random.random(
            size=(len(f_s_coefficients_flattened), len(f_s_coefficients_flattened))
        )
        random_sensing_measurements = random_sensing_matrix.dot(
            f_s_coefficients_flattened
        )
        f_s_coefficients_estimate = np.linalg.pinv(random_sensing_matrix).dot(
            random_sensing_measurements
        )
        assert np.allclose(f_s_coefficients_estimate, f_s_coefficients_flattened)

    # def test_random_sensing_of_f_s_coefficients(self):
    #     t_k_1 = np.array([1,1.5])
    #     c_k_1 = np.array([1,-1])
    #     c_k_2 = np.array([0.5,3])
    #     fri_signal_1 = src.FRISignal.FRISignal(t_k_1, c_k_1, 2.0)
    #     fri_signal_2 = src.FRISignal.FRISignal(t_k_1, c_k_2, 2.0)
    #     M = 3
    #     f_s_coefficients = np.zeros((2,2*M+1), dtype = 'complex')
    #     f_s_coefficients[0,:] = fri_signal_1.get_fourier_series(np.arange(-M, M+1,1).T)
    #     f_s_coefficients[1,:] = fri_signal_2.get_fourier_series(np.arange(-M, M+1,1).T)
    #     f_s_coefficients_flattened = f_s_coefficients.flatten()
    #     num_constraints = int(len(f_s_coefficients_flattened)/4)*4
    #     num_coefficients = len(f_s_coefficients_flattened)
    #
    #     random_sensing_matrix = np.zeros((num_constraints, num_coefficients), dtype = 'complex')
    #     random_sensing_matrix[:int(num_constraints/4),:int(num_coefficients/2)] = np.random.random(size = (int(num_constraints/4),int(num_coefficients/2)))
    #     random_sensing_matrix[int(num_constraints/4):int(num_constraints/2),:int(num_coefficients/2)] = random_sensing_matrix[:int(num_constraints/4),int(num_coefficients/2)-1::-1].conj()
    #     random_sensing_matrix[int(num_constraints/2):int(num_constraints*3/4),int(num_coefficients/2):] = np.random.random(size = (int(num_constraints/4),int(num_coefficients/2)))
    #     random_sensing_matrix[int(num_constraints*3/4):,int(num_coefficients/2):] = random_sensing_matrix[int(num_constraints/2):int(num_constraints*3/4),-1:int(num_coefficients/2)-1:-1].conj()
    #     random_sensing_measurements = random_sensing_matrix.dot(f_s_coefficients_flattened)
    #
    #     f_s_coefficients_estimate = np.linalg.pinv(random_sensing_matrix).dot(random_sensing_measurements)
    #     distance_0 = np.linalg.norm(f_s_coefficients_estimate - f_s_coefficients_flattened)
    #     assert not np.allclose(f_s_coefficients_estimate, f_s_coefficients_flattened)
    #     print(distance_0)
    #
    #     for i in range(30):
    #         f_s_coefficients_estimate_reshaped = np.reshape(f_s_coefficients_estimate,(2,-1))
    #         a_filter = src.FRISignal.AnnihilatingFilter(f_s_coefficients_estimate_reshaped)
    #         print(a_filter.get_filter_coefficients())
    #         f_s_coefficients_post_annihilation = np.zeros((2,2*M+1), dtype = 'complex')
    #         f_s_coefficients_post_annihilation[0,:] = a_filter.make_annihilatable(f_s_coefficients_estimate_reshaped[0,:])
    #         f_s_coefficients_post_annihilation[1,:] = a_filter.make_annihilatable(f_s_coefficients_estimate_reshaped[1,:])
    #         distance_1 = np.linalg.norm(f_s_coefficients_post_annihilation.flatten() - f_s_coefficients_flattened)
    #         print(distance_1)
    #
    #         error = random_sensing_measurements-random_sensing_matrix.dot(f_s_coefficients_post_annihilation.flatten())
    #         # print(error)
    #         f_s_coefficients_estimate = f_s_coefficients_post_annihilation.flatten() + np.linalg.pinv(random_sensing_matrix).dot(error)
    #         distance_2 = np.linalg.norm(f_s_coefficients_estimate - f_s_coefficients_flattened)
    #         print(distance_2)
    #         # print(f_s_coefficients_estimate)
    #     assert False

    # def test_rank_is_M_minus_1(self):
    #     t_k = np.array([1,1.5])
    #     c_k_1 = np.array([1,-1])
    #     c_k_2 = np.array([0.5,3])
    #     fri_signal_1 = src.FRISignal.FRISignal(t_k, c_k_1, 2.0)
    #     fri_signal_2 = src.FRISignal.FRISignal(t_k, c_k_2, 2.0)
    #     M = 3
    #     f_s_coefficients = np.zeros((2,2*M+1), dtype = 'complex')
    #     f_s_coefficients[0,:] = fri_signal_1.get_fourier_series(np.arange(-M, M+1,1).T)
    #     f_s_coefficients[1,:] = fri_signal_2.get_fourier_series(np.arange(-M, M+1,1).T)
    #     a_filter = src.FRISignal.AnnihilatingFilter(f_s_coefficients)
    #     result_1 = np.convolve(a_filter.get_filter_coefficients(), f_s_coefficients[0,:])
    #     result_2 = np.convolve(a_filter.get_filter_coefficients(), f_s_coefficients[1,:])
    #     assert np.allclose(result_1[M:-M], np.zeros_like(result_1[M:-M]))
    #     assert np.allclose(result_2[M:-M], np.zeros_like(result_2[M:-M]))
    #     # TODO create matrix that should be of rank -1 and check that it holds.
    #     assert False

    def test_can_find_annihilator_4_signals(self):
        K = 10
        M = 9
        t_k_1 = np.random.uniform(low=0.0, high=2.0, size=(10))

        c_k_1 = np.random.uniform(low=-2.0, high=2.0, size=(10))
        c_k_2 = np.random.uniform(low=-2.0, high=2.0, size=(10))
        c_k_3 = np.random.uniform(low=-2.0, high=2.0, size=(10))
        c_k_4 = np.random.uniform(low=-2.0, high=2.0, size=(10))
        fri_signal_1 = src.FRISignal.FRISignal(t_k_1, c_k_1, 2.0)
        fri_signal_2 = src.FRISignal.FRISignal(t_k_1, c_k_2, 2.0)
        fri_signal_3 = src.FRISignal.FRISignal(t_k_1, c_k_3, 2.0)
        fri_signal_4 = src.FRISignal.FRISignal(t_k_1, c_k_4, 2.0)

        M = 9
        f_s_coefficients = np.zeros((4, 2 * M + 1), dtype="complex")
        f_s_coefficients[0, :] = fri_signal_1.get_fourier_series(
            np.arange(-M, M + 1, 1).T
        )
        f_s_coefficients[1, :] = fri_signal_2.get_fourier_series(
            np.arange(-M, M + 1, 1).T
        )
        f_s_coefficients[2, :] = fri_signal_3.get_fourier_series(
            np.arange(-M, M + 1, 1).T
        )
        f_s_coefficients[3, :] = fri_signal_4.get_fourier_series(
            np.arange(-M, M + 1, 1).T
        )

        filter_length = K + 1
        a_filter = src.FRISignal.AnnihilatingFilter(f_s_coefficients, filter_length)
        print(a_filter.get_filter_coefficients())

        K_s = np.atleast_2d(np.arange(0.0, filter_length - 0.1, 1.0)).T
        target_coefficients = np.product(
            1
            - np.exp(-1j * 2 * np.pi * t_k_1 / 2.0)
            * np.exp(-1j * 2 * np.pi * K_s / filter_length),
            1,
        )
        print("TARGET: ", target_coefficients)
        print("RESULT: ", np.fft.fft(a_filter.get_filter_coefficients()))

        assert np.allclose(
            target_coefficients, np.fft.fft(a_filter.get_filter_coefficients())
        )

    # def test_can_find_annihilator_single_signal_w_constraints_only(self):
    #     num_diracs = 10
    #     num_signals = 5
    #     t_k = 2*np.random.random(size = (num_diracs))
    #     c_k_1 = 2*(np.random.random(size = (num_diracs)) - 0.5)
    #     c_k_2 = 2*(np.random.random(size = (num_diracs)) - 0.5)
    #
    #     M = num_diracs + 1
    #     c_k = []
    #     fri_signals = []
    #     f_s_coefficients = []
    #     random_sensing_matrices = []
    #     measurements = []
    #     for n_s in range(num_signals):
    #         c_k.append(2*(np.random.random(size = (num_diracs)) - 0.5))
    #         fri_signals.append(src.FRISignal.FRISignal(t_k, c_k[n_s], 2.0))
    #         f_s_coefficients.append(fri_signals[n_s].get_fourier_series(np.arange(-M, M+1,1).T))
    #         random_sensing_matrices.append(np.random.random(size = (len(f_s_coefficients[n_s].flatten())-1, len(f_s_coefficients[n_s].flatten()))))
    #         measurements.append(random_sensing_matrices[n_s].dot(f_s_coefficients[n_s]))
    #
    #     recovered_circ_matrix = self.recover_from_meas(random_sensing_matrices, measurements,num_diracs, num_signals)
    #
    #     # a_filter = src.FRISignal.AnnihilatingFilter(f_s_coefficients)
    #     # print(a_filter.get_filter_coefficients())
    #     # result = np.convolve(a_filter.get_filter_coefficients(), f_s_coefficients)
    #     # assert np.allclose(result[M-1:-M+1:1], np.zeros_like(result[M-1:-M+1:1]))
    #     print("FS 1: ", self.get_circular_matrix(f_s_coefficients[0], num_diracs))
    #     print("FS 2: ", self.get_circular_matrix(f_s_coefficients[1], num_diracs))
    #     print("recovered: ", recovered_circ_matrix)
    #     stacked_circular_matrices = self.get_circular_matrix(f_s_coefficients[0], num_diracs)
    #     for n_s in range(1, num_signals):
    #         stacked_circular_matrices = np.concatenate((stacked_circular_matrices, self.get_circular_matrix(f_s_coefficients[n_s], num_diracs)))
    #     assert np.allclose(recovered_circ_matrix.flatten(), stacked_circular_matrices.flatten())

    # assert False

    def get_circular_matrix(self, hidden_f_s_coefficients, num_diracs: int):
        circular_matrix_shape = (num_diracs + 2, num_diracs + 2)
        circular_matrix = np.zeros(circular_matrix_shape, dtype="complex")
        np.set_printoptions(linewidth=np.inf)
        # print(hidden_f_s_coefficients)

        for i in range(num_diracs + 2):
            # has been checked
            circular_matrix[i, :] = hidden_f_s_coefficients[i : num_diracs + i + 2][
                ::-1
            ]
        return circular_matrix

    def recover_from_meas(
        self,
        random_sensing_matrices: list,
        measurements: list,
        num_diracs: int,
        num_signals: int,
    ):
        assert len(random_sensing_matrices) == num_signals
        assert len(measurements) == num_signals
        recovered_matrix_shape = (num_signals * (num_diracs + 2), num_diracs + 2)
        n_circularity_constraint_per_signal = (num_diracs + 2) * (num_diracs + 1) - 1
        circularity_constraints = np.zeros(
            (
                num_signals * n_circularity_constraint_per_signal,
                np.product(recovered_matrix_shape),
            )
        )
        # This has been verified already
        for n_s in range(num_signals):
            for i in range(num_diracs + 1):
                for j in range(1, num_diracs + 2):
                    circularity_constraints[
                        n_s * n_circularity_constraint_per_signal
                        + i * (num_diracs + 2)
                        + j
                        - 1,
                        n_s * (num_diracs + 2) ** 2 + (i + 1) * (num_diracs + 2) + j,
                    ] = 1
                    circularity_constraints[
                        n_s * n_circularity_constraint_per_signal
                        + i * (num_diracs + 2)
                        + j
                        - 1,
                        n_s * (num_diracs + 2) ** 2 + i * (num_diracs + 2) + j - 1,
                    ] = -1
        concatenated_measurements = np.zeros((circularity_constraints.shape[0],))

        projection_constraints = np.zeros(
            (0, np.product(recovered_matrix_shape)), dtype="complex"
        )
        # print(projection_constraints.shape)
        for n_s_m in range(num_signals):
            random_sensing_matrix = random_sensing_matrices[n_s_m]
            projection_constraints = np.concatenate(
                (
                    np.zeros(
                        (
                            random_sensing_matrix.shape[0],
                            np.product(recovered_matrix_shape),
                        ),
                        dtype="complex",
                    ),
                    projection_constraints,
                )
            )
            for i in range(random_sensing_matrix.shape[0]):
                # has been checked
                projection_constraints[
                    i,
                    n_s_m * (num_diracs + 2) ** 2
                    + 0 : n_s_m * (num_diracs + 2) ** 2
                    + num_diracs
                    + 2,
                ] = random_sensing_matrix[i, 0 : num_diracs + 2][::-1]
                projection_constraints[
                    i,
                    (n_s_m + 1) * (num_diracs + 2) ** 2
                    - (num_diracs + 2) : (n_s_m + 1) * (num_diracs + 2) ** 2
                    - 1,
                ] = random_sensing_matrix[i, num_diracs + 2 : :][::-1]

            concatenated_measurements = np.concatenate(
                (measurements[n_s_m], concatenated_measurements)
            )
        concatenated_constraints = np.concatenate(
            (projection_constraints, circularity_constraints)
        )
        circ_matrix_recovered = src.helpers.singular_value_projection_w_matrix(
            recovered_matrix_shape,
            concatenated_constraints,
            concatenated_measurements,
            num_diracs + 1,
            tol=1e-3,
            lr=0.5,
        )
        # print("recovered", circ_matrix_recovered, "end")

        return circ_matrix_recovered
