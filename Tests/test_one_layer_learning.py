import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/..")
from src import *


class TestLearningWorksOneSignal:
    def test_one_signal_2_by_2(self):
        int_shift = [-1, -0.1]

        np.random.seed(10)
        original1 = src.signals.periodicBandlimitedSignal(
            10, 10, np.random.random(size=(10,))
        )
        np.random.seed(11)
        original2 = src.signals.periodicBandlimitedSignal(
            10, 10, np.random.random(size=(10,))
        )
        signals = src.signals.periodicBandlimitedSignals(10)
        signals.add(original1)
        signals.add(original2)
        A = [[0.9, 0.1], [0.2, 0.8]]

        tem_params = TEMParams(1, 1, 1, A, int_shift)

        spikes_mult = encoder.ContinuousEncoder(tem_params).encode(signals, 7)
        print(spikes_mult.get_total_num_spike_diffs())
        print(spikes_mult)

        single_layer = Layer(2, 2)
        single_layer.learn_weight_matrix_from_one_ex(signals, spikes_mult)
        print(single_layer.weight_matrix)
        print(A)
        assert np.allclose(single_layer.weight_matrix, A, atol=1e-2, rtol=1e-2)

    def test_one_signal_5_by_2(self):
        int_shift = [-1, -0.1]

        np.random.seed(10)
        original1 = src.signals.periodicBandlimitedSignal(
            10, 10, np.random.random(size=(10,))
        )
        np.random.seed(11)
        original2 = src.signals.periodicBandlimitedSignal(
            10, 10, np.random.random(size=(10,))
        )
        signals = src.signals.periodicBandlimitedSignals(10)
        signals.add(original1)
        signals.add(original2)
        A = np.random.random(size=(5, 2))

        tem_params = TEMParams(1, 1, 1, A)

        spikes_mult = encoder.ContinuousEncoder(tem_params).encode(signals, 6)
        print(spikes_mult.get_total_num_spike_diffs())

        single_layer = Layer(2, 5)
        single_layer.learn_weight_matrix_from_one_ex(signals, spikes_mult)

        assert np.allclose(single_layer.weight_matrix, A, atol=1e-2, rtol=1e-2)


class TestLearningWorksMultiSignal:
    def test_multi_signal_2_by_2(self):
        int_shift = [-1, -0.1]

        np.random.seed(10)
        signals_1 = src.signals.periodicBandlimitedSignals(10)
        signals_1.add(
            src.signals.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )
        signals_1.add(
            src.signals.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )
        A = [[0.9, 0.1], [0.2, 0.8]]

        signals_2 = src.signals.periodicBandlimitedSignals(10)
        signals_2.add(
            src.signals.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )
        signals_2.add(
            src.signals.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )

        tem_params = TEMParams(1, 1, 1, A, int_shift)

        spikes_mult_1 = encoder.ContinuousEncoder(tem_params).encode(signals_1, 3.5)
        spikes_mult_2 = encoder.ContinuousEncoder(tem_params).encode(signals_2, 3.5)

        single_layer = Layer(2, 2)
        single_layer.learn_weight_matrix_from_m_ex(
            [signals_1, signals_2], [spikes_mult_1, spikes_mult_2]
        )

        assert np.allclose(A, single_layer.weight_matrix, atol=1e-2, rtol=1e-2)

    def test_multi_signal_5_by_2(self):
        int_shift = [-1, -0.1]

        np.random.seed(10)

        np.random.seed(10)
        signals_1 = src.signals.periodicBandlimitedSignals(10)
        signals_1.add(
            src.signals.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )
        signals_1.add(
            src.signals.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )
        A = np.random.random(size=(5, 2))

        signals_2 = src.signals.periodicBandlimitedSignals(10)
        signals_2.add(
            src.signals.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )
        signals_2.add(
            src.signals.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )

        tem_params = TEMParams(1, 1, 1, A)

        spikes_mult_1 = encoder.ContinuousEncoder(tem_params).encode(signals_1, 3)
        spikes_mult_2 = encoder.ContinuousEncoder(tem_params).encode(signals_2, 3)

        single_layer = Layer(2, 5)
        single_layer.learn_weight_matrix_from_m_ex(
            [signals_1, signals_2], [spikes_mult_1, spikes_mult_2]
        )
        print(A)
        print(single_layer.weight_matrix)
        assert np.allclose(single_layer.weight_matrix, A, atol=1e-2, rtol=1e-2)
