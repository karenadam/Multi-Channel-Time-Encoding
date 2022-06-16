import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/..")
from src import *


class TestLearningWorksOneExample:
    def test_2_by_2_can_recover_f_s_coefficients(self):
        int_shift = [-1, -0.1]
        period = 10
        n_components = 10

        np.random.seed(10)
        signals = SignalCollection.periodicBandlimitedSignals(period)
        signals.add(
            Signal.periodicBandlimitedSignal(
                period, n_components, np.random.random(size=(10,))
            )
        )
        signals.add(
            Signal.periodicBandlimitedSignal(
                period, n_components, np.random.random(size=(10,))
            )
        )
        A = [[0.9, 0.1], [0.2, 0.8]]

        tem_params = TEMParams(1, 1, 1, A, int_shift)

        spikes_mult = encoder.ContinuousEncoder(tem_params).encode(signals, 13)
        tem_params.mixing_matrix = np.eye(2)
        print(spikes_mult.get_total_num_spike_diffs())

        single_layer = Layer(2, 2)
        recovered_f_s_coefficients = single_layer.get_preactivation_fsc(
            spikes_mult, 10, period, real_f_s=True
        )
        assert np.allclose(
            recovered_f_s_coefficients,
            np.array(A).dot(np.array(signals.coefficient_values)),
            atol=1e-3,
        )

    def test_2_by_2_can_recover_complex_f_s_coefficients(self):
        int_shift = [-1, -0.1]
        period = 10
        n_components = 10

        np.random.seed(10)
        signals = SignalCollection.periodicBandlimitedSignals(period)
        f_s_components_signal_1 = np.random.random(size=(10,)) + 1j * np.random.random(
            size=(10,)
        )
        f_s_components_signal_1[0] = np.real(f_s_components_signal_1[0])
        signals.add(
            Signal.periodicBandlimitedSignal(
                period, n_components, f_s_components_signal_1
            )
        )

        f_s_components_signal_2 = np.random.random(size=(10,)) + 1j * np.random.random(
            size=(10,)
        )
        f_s_components_signal_2[0] = np.real(f_s_components_signal_2[0])
        signals.add(
            Signal.periodicBandlimitedSignal(
                period, n_components, f_s_components_signal_2
            )
        )
        A = [[0.9, 0.1], [0.2, 0.8]]

        tem_params = TEMParams(1, 1, 1, A, int_shift)

        spikes_mult = encoder.ContinuousEncoder(tem_params).encode(signals, 30)
        tem_params.mixing_matrix = np.eye(2)
        print(spikes_mult.get_total_num_spike_diffs())

        single_layer = Layer(2, 2)
        recovered_f_s_coefficients = single_layer.get_preactivation_fsc(
            spikes_mult, 10, period, real_f_s=False
        )
        print(recovered_f_s_coefficients)
        print(np.array(A).dot(np.array(signals.coefficient_values)))
        assert np.allclose(
            recovered_f_s_coefficients,
            np.array(A).dot(np.array(signals.coefficient_values)),
            atol=1e-2,
        )

    def test_2_by_2_can_find_fri_f_s_coeffs(self):
        int_shift = [-1, -0.1]
        period = 3.5
        n_components = 8

        K = 8
        num_diracs_per_signal = 4
        t_k_1 = np.random.uniform(low=0.0, high=period, size=(num_diracs_per_signal))
        t_k_2 = np.random.uniform(low=0.0, high=period, size=(num_diracs_per_signal))
        t_k = np.concatenate((t_k_1, t_k_2))

        fri_signal_1 = FRISignal.FRISignal(t_k_1, np.ones_like(t_k_1), period)
        fri_signal_2 = FRISignal.FRISignal(t_k_2, np.ones_like(t_k_2), period)

        f_s_components_signal_1 = fri_signal_1.get_fourier_series(
            np.arange(0, n_components, 1).T
        )
        f_s_components_signal_2 = fri_signal_2.get_fourier_series(
            np.arange(0, n_components, 1).T
        )

        np.random.seed(10)
        signals = SignalCollection.periodicBandlimitedSignals(period)
        signals.add(
            Signal.periodicBandlimitedSignal(
                period, n_components, f_s_components_signal_1
            )
        )
        signals.add(
            Signal.periodicBandlimitedSignal(
                period, n_components, f_s_components_signal_2
            )
        )
        A = [[0.9, 0.1], [0.2, 0.8]]

        tem_params = TEMParams(1, 1, 1, A, int_shift)

        spikes_mult = encoder.ContinuousEncoder(tem_params).encode(signals, 50)
        tem_params.mixing_matrix = np.eye(2)

        single_layer = Layer(2, 2)
        recovered_f_s_coefficients = single_layer.get_preactivation_fsc(
            spikes_mult, n_components, period, real_f_s=False
        )

        assert np.allclose(
            recovered_f_s_coefficients,
            np.array(A).dot(np.array(signals.coefficient_values)),
            atol=1e-2,
        )

    def test_2_by_2_can_find_spike_times(self):
        int_shift = [-1, -0.1]
        period = 3.5
        n_components = 8

        K = 8
        num_diracs_per_signal = 4
        t_k_1 = np.random.uniform(low=0.0, high=period, size=(num_diracs_per_signal))
        t_k_2 = np.random.uniform(low=0.0, high=period, size=(num_diracs_per_signal))
        t_k = np.concatenate((t_k_1, t_k_2))

        fri_signal_1 = FRISignal.FRISignal(t_k_1, np.ones_like(t_k_1), period)
        fri_signal_2 = FRISignal.FRISignal(t_k_2, np.ones_like(t_k_2), period)

        f_s_components_signal_1 = fri_signal_1.get_fourier_series(
            np.arange(0, n_components, 1).T
        )
        f_s_components_signal_2 = fri_signal_2.get_fourier_series(
            np.arange(0, n_components, 1).T
        )

        np.random.seed(10)
        signals = SignalCollection.periodicBandlimitedSignals(period)
        signals.add(
            Signal.periodicBandlimitedSignal(
                period, n_components, f_s_components_signal_1
            )
        )
        signals.add(
            Signal.periodicBandlimitedSignal(
                period, n_components, f_s_components_signal_2
            )
        )
        A = [[0.9, 0.1], [0.2, 0.8]]

        tem_params = TEMParams(1, 1, 1, A, int_shift)

        spikes_mult = encoder.ContinuousEncoder(tem_params).encode(signals, 50)
        tem_params.mixing_matrix = np.eye(2)

        single_layer = Layer(2, 2)
        spike_times = single_layer.learn_spike_input_and_weight_matrix_from_one_example(
            spikes_mult, n_components, period
        )
        assert np.allclose(np.sort(spike_times), np.sort(t_k), atol=1e-2)

    # def test_4_signals_can_find_spike_times(self):
    #     period = 3.5
    #
    #     num_diracs_per_signal = 4
    #     t_k_1 = np.random.uniform(low=0.0, high=period, size=(num_diracs_per_signal))
    #     t_k_2 = np.random.uniform(low=0.0, high=period, size=(num_diracs_per_signal))
    #     t_k_3 = np.random.uniform(low=0.0, high=period, size=(num_diracs_per_signal))
    #     t_k_4 = np.random.uniform(low=0.0, high=period, size=(num_diracs_per_signal))
    #     t_k = np.concatenate((t_k_1, t_k_2, t_k_3, t_k_4))
    #
    #     fri_signal_1 = src.signals.FRISignal(t_k_1, np.ones_like(t_k_1), period)
    #     fri_signal_2 = src.signals.FRISignal(t_k_2, np.ones_like(t_k_2), period)
    #     fri_signal_3 = src.signals.FRISignal(t_k_3, np.ones_like(t_k_3), period)
    #     fri_signal_4 = src.signals.FRISignal(t_k_4, np.ones_like(t_k_4), period)
    #
    #     n_f_s = num_diracs_per_signal + 1
    #     f_s_coefficients = np.zeros((4, 2 * n_f_s - 1), dtype="complex")
    #     f_s_coefficients[0, :] = fri_signal_1.get_fourier_series(
    #         np.arange(-n_f_s + 1, n_f_s, 1).T
    #     )
    #     f_s_coefficients[1, :] = fri_signal_2.get_fourier_series(
    #         np.arange(-n_f_s + 1, n_f_s, 1).T
    #     )
    #     f_s_coefficients[2, :] = fri_signal_3.get_fourier_series(
    #         np.arange(-n_f_s + 1, n_f_s, 1).T
    #     )
    #     f_s_coefficients[3, :] = fri_signal_4.get_fourier_series(
    #         np.arange(-n_f_s + 1, n_f_s, 1).T
    #     )
    #
    #     filter_length = 4 * num_diracs_per_signal + 1
    #     a_filter = src.FRISignal.AnnihilatingFilter(f_s_coefficients, filter_length)
    #     print("filter: ", np.real(a_filter.get_filter_coefficients()))
    #     filter_poly = np.polynomial.polynomial.Polynomial(
    #         a_filter.get_filter_coefficients()
    #     )
    #     roots = filter_poly.roots()
    #     recovered_times = np.sort(
    #         np.mod(np.angle(roots) * period / (2 * np.pi), period)
    #     )
    #     print(a_filter.get_filter_coefficients())
    #     print("HERE", recovered_times)
    #     print(np.sort(t_k))
    #     assert np.allclose(recovered_times, np.sort(t_k), atol=1e-1)

    def test_2_by_2_can_find_weights_multi(self):
        int_shift = [-1, -0.1]
        period = 3.5
        n_components = 8

        K = 8
        num_diracs_per_signal = 4
        np.random.seed(53)

        spikes_mult = []

        n_examples = 2
        t_k = []
        for n_e in range(n_examples):

            t_k_1 = np.random.uniform(
                low=0.0, high=period, size=(num_diracs_per_signal)
            )
            t_k_2 = np.random.uniform(
                low=0.0, high=period, size=(num_diracs_per_signal)
            )
            t_k.append(np.concatenate((t_k_1, t_k_2)))

            fri_signal_1 = FRISignal.FRISignal(t_k_1, np.ones_like(t_k_1), period)
            fri_signal_2 = FRISignal.FRISignal(t_k_2, np.ones_like(t_k_2), period)

            f_s_components_signal_1 = fri_signal_1.get_fourier_series(
                np.arange(0, n_components, 1).T
            )
            f_s_components_signal_2 = fri_signal_2.get_fourier_series(
                np.arange(0, n_components, 1).T
            )

            signals = SignalCollection.periodicBandlimitedSignals(period)
            signals.add(
                Signal.periodicBandlimitedSignal(
                    period, n_components, f_s_components_signal_1
                )
            )
            signals.add(
                Signal.periodicBandlimitedSignal(
                    period, n_components, f_s_components_signal_2
                )
            )
            A = [[0.9, 0.1], [0.2, 0.8]]

            tem_params = TEMParams(1, 1, 1, A, int_shift)

            spikes_mult.append(
                encoder.ContinuousEncoder(tem_params).encode(signals, 50)
            )

        single_layer = Layer(2, 2)
        spike_times = (
            single_layer.learn_spike_input_and_weight_matrix_from_multi_example(
                spikes_mult, n_components, period
            )
        )

        print(single_layer.weight_matrix)
        for n_e in range(n_examples):
            t_k_hat = np.sort(
                [item for sublist in spike_times[n_e] for item in sublist]
            )
            print(t_k_hat)
            print(np.sort(t_k[n_e]))
            assert np.allclose(t_k_hat, np.sort(t_k[n_e]), atol=1e-1)

        # Need to find a way to merge two coinciding times together, otherwise comparison is not fair..
        # and to compare weight matrices
        # assert np.allclose(np.sort(spike_times), np.sort(t_k), atol=1e-2)
