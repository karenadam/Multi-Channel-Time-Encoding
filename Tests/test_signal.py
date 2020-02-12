import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../Source")
from Time_Encoder import timeEncoder
from Signal import *

import scipy
import scipy.signal


class TestBandlimitedSignal:
    def test_integral(self):
        delta_t = 0.1
        t = np.arange(0, 10, delta_t)
        sinc_locs = [1.3, 2.8, 7.2]
        sinc_amps = [1, 5, 3]
        Omega = np.pi
        signal = bandlimitedSignal(Omega, sinc_locs, sinc_amps)
        t_int_0 = np.arange(2, 7, 0.01)
        t_int_1 = np.arange(4, 8, 0.01)
        discrete_integral_0 = signal.get_total_integral(t_int_0)
        discrete_integral_1 = signal.get_total_integral(t_int_1)
        precise_integral = signal.get_precise_integral([2, 4], [7, 8])

        assert (
            np.abs(discrete_integral_0 - precise_integral[0])
            < np.abs(discrete_integral_0) * 1e-2
        )
        assert (
            np.abs(discrete_integral_1 - precise_integral[1])
            < np.abs(discrete_integral_1) * 1e-2
        )


class TestBandlimitedSignals:
    def test_integral(self):
        sinc_locs = [1, 2, 3]
        sinc_amps = [[3, 2, 6], [1, -2, 4]]
        omega = np.pi
        signals = bandlimitedSignals(omega, sinc_locs, sinc_amps)
        t_start = [[0, 1.5, 3], [0.5, 2]]
        t_end = [[1.5, 3, 4], [2, 3]]
        integrals = signals.get_integrals(t_start, t_end)

        signal_1 = signals.get_signal(1)
        signal_1_integrals = signal_1.get_precise_integral(t_start[1], t_end[1])

        assert np.linalg.norm(integrals[3:] - signal_1_integrals) < 1e-6

    def test_mixing(self):
        sinc_locs = [1, 2, 3]
        sinc_amps = [[1, 0, 1], [1, 1, 0]]
        omega = np.pi
        signals = bandlimitedSignals(omega, sinc_locs, sinc_amps)
        mixed_amplitudes = signals.mix_amplitudes([[2, 1], [1, 0]])
        expected_amplitudes = [3, 1, 2, 1, 0, 1]

        assert np.linalg.norm(mixed_amplitudes - expected_amplitudes) < 1e-6


class TestPiecewiseConstantSignal:
    def test_sampling(self):
        discontinuities = [1, 2]
        values = [1]
        signal = piecewiseConstantSignal(discontinuities, values)
        samples = signal.sample([0, 0.1, 1.2, 1.5, 1.7, 2.3, 3.5])
        assert samples[0] == 0
        assert samples[1] == 0
        assert samples[2] == 1
        assert samples[3] == 1
        assert samples[4] == 1
        assert samples[5] == 0
        assert samples[6] == 0

    def test_sampling2(self):
        discontinuities = [1, 2, 5, 6]
        values = [1, 4, -2]
        signal = piecewiseConstantSignal(discontinuities, values)
        samples = signal.sample([0, 0.1, 1.2, 1.5, 1.7, 2.3, 3.5, 5.6, 6.5])
        assert samples[0] == 0
        assert samples[1] == 0
        assert samples[2] == 1
        assert samples[3] == 1
        assert samples[4] == 1
        assert samples[5] == 4
        assert samples[6] == 4
        assert samples[7] == -2
        assert samples[8] == 0


class TestPiecewiseConstantSignals:
    def test_creation(self):
        discontinuities = [[1, 2], [1.5, 4]]
        values = [[1], [-2]]
        signals = piecewiseConstantSignals(discontinuities, values)
        signal_1 = signals.get_signal(1)
        assert signal_1.get_discontinuities() == [1.5, 4]
        assert signal_1.get_values() == [-2]

    def test_sampling(self):
        discontinuities = [[1, 2, 4, 6, 7, 8], [1.5, 4, 7, 9, 10]]
        values = [[1, 3, 4, -2, 3], [-2, 1, 3, -1]]
        signals = piecewiseConstantSignals(discontinuities, values)
        signal_1 = signals.get_signal(1)

        sample_locs = [0, 1.5, 3]
        samples_1_matrix_approach = signals.sample(sample_locs, np.pi)[3:].T
        samples_1_loop_approach = signal_1.low_pass_filter(np.pi).sample(
            np.array(sample_locs)
        )

        assert (
            np.linalg.norm(samples_1_matrix_approach - samples_1_loop_approach) < 1e-6
        )


class TestLPFPCSsignal:
    def test_sampling(self):
        omega = np.pi
        discontinuities = [1, 2, 5, 6]
        values = [1, 4, -2]
        t = np.arange(0, 10, 0.1)
        signal = piecewiseConstantSignal(discontinuities, values)
        samples = signal.sample(t)
        filtered_signal = signal.low_pass_filter(omega)
        samples_filtered = filtered_signal.sample(t)

        sampled_sinc = sinc(t - 5, omega)
        samples_discrete_filtered = scipy.signal.convolve(samples, sampled_sinc) * (
            t[1] - t[0]
        )
        offset = int(len(sampled_sinc) / 2)

        assert (
            np.abs(samples_filtered[40] - samples_discrete_filtered[offset + 40]) < 1e-2
        )
