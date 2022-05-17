import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/..")
from src import *


class TestLearningWorksOneSignal:
    def test_one_signal_2_by_2(self):
        int_shift = [-1, -0.1]

        np.random.seed(10)
        original1 = Signal.periodicBandlimitedSignal(
            10, 10, np.random.random(size=(10,))
        )
        np.random.seed(11)
        original2 = Signal.periodicBandlimitedSignal(
            10, 10, np.random.random(size=(10,))
        )
        signals = Signal.periodicBandlimitedSignals(10)
        signals.add(original1)
        signals.add(original2)
        A = [[0.9, 0.1], [0.2, 0.8]]

        tem_params = TEMParams(1, 1, 1, A, int_shift)

        spikes_mult = Encoder.ContinuousEncoder(tem_params).encode(signals, 5)
        print(spikes_mult.get_total_num_spike_diffs())

        single_layer = Layer(2, 2)
        single_layer.learn_weight_matrix_from_one_signal(signals, spikes_mult)

        assert np.allclose(single_layer.weight_matrix, A)

    def test_one_signal_5_by_2(self):
        int_shift = [-1, -0.1]

        np.random.seed(10)
        original1 = Signal.periodicBandlimitedSignal(
            10, 10, np.random.random(size=(10,))
        )
        np.random.seed(11)
        original2 = Signal.periodicBandlimitedSignal(
            10, 10, np.random.random(size=(10,))
        )
        signals = Signal.periodicBandlimitedSignals(10)
        signals.add(original1)
        signals.add(original2)
        A = np.random.random(size=(5, 2))

        tem_params = TEMParams(1, 1, 1, A)

        spikes_mult = Encoder.ContinuousEncoder(tem_params).encode(signals, 6)
        print(spikes_mult.get_total_num_spike_diffs())

        single_layer = Layer(2, 5)
        single_layer.learn_weight_matrix_from_one_signal(signals, spikes_mult)

        assert np.allclose(single_layer.weight_matrix, A)


class TestLearningWorksMultiSignal:
    def test_multi_signal_2_by_2(self):
        int_shift = [-1, -0.1]

        np.random.seed(10)
        signals_1 = Signal.periodicBandlimitedSignals(10)
        signals_1.add(
            Signal.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )
        signals_1.add(
            Signal.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )
        A = [[0.9, 0.1], [0.2, 0.8]]

        signals_2 = Signal.periodicBandlimitedSignals(10)
        signals_2.add(
            Signal.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )
        signals_2.add(
            Signal.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )

        tem_params = TEMParams(1, 1, 1, A, int_shift)

        spikes_mult_1 = Encoder.ContinuousEncoder(tem_params).encode(signals_1, 3)
        spikes_mult_2 = Encoder.ContinuousEncoder(tem_params).encode(signals_2, 3)

        single_layer = Layer(2, 2)
        single_layer.learn_weight_matrix_from_multi_signals(
            [signals_1, signals_2], [spikes_mult_1, spikes_mult_2]
        )

        assert np.allclose(single_layer.weight_matrix, A)

    def test_multi_signal_5_by_2(self):
        int_shift = [-1, -0.1]

        np.random.seed(10)

        np.random.seed(10)
        signals_1 = Signal.periodicBandlimitedSignals(10)
        signals_1.add(
            Signal.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )
        signals_1.add(
            Signal.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )
        A = np.random.random(size=(5, 2))

        signals_2 = Signal.periodicBandlimitedSignals(10)
        signals_2.add(
            Signal.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )
        signals_2.add(
            Signal.periodicBandlimitedSignal(10, 10, np.random.random(size=(10,)))
        )

        tem_params = TEMParams(1, 1, 1, A)

        spikes_mult_1 = Encoder.ContinuousEncoder(tem_params).encode(signals_1, 3)
        spikes_mult_2 = Encoder.ContinuousEncoder(tem_params).encode(signals_2, 3)

        single_layer = Layer(2, 5)
        single_layer.learn_weight_matrix_from_multi_signals(
            [signals_1, signals_2], [spikes_mult_1, spikes_mult_2]
        )

        assert np.allclose(single_layer.weight_matrix, A)
