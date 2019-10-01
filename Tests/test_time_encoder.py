import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../Source")
from Time_Encoder import timeEncoder
from Signal import bandlimitedSignal


class TestTimeEncoderSingleSignalSingleChannel:
    def test_time_encoder_creator(self):
        TEM = timeEncoder(kappa=1, delta=1, b=1)
        assert TEM.kappa == [1], "kappa was falsely assigned"
        assert TEM.delta == [1], "delta was falsely assigned"
        assert TEM.b == [1], "b was falsely assigned"

        TEM2 = timeEncoder(kappa=[1], delta=1, b=1, n_channels=2)
        assert len(TEM2.kappa) == 2

    def test_can_reconstruct_standard_encoding(self):
        kappa = 1
        delta = 1

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 15, delta_t)
        original = bandlimitedSignal(t, delta_t, omega, seed=10)
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1

        tem_single = timeEncoder(kappa, delta, b)
        spikes_single = tem_single.encode(y, delta_t)
        rec_single = tem_single.decode(spikes_single, t, omega, delta_t)
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)
        assert (
            np.mean(((rec_single - y) ** 2)[start_index:end_index]) / np.mean(y ** 2)
            < 1e-3
        )

    def test_can_reconstruct_precise_encoding(self):
        kappa = 1
        delta = 1

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 15, delta_t)
        original = bandlimitedSignal(t, delta_t, omega, seed=10)
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1

        tem_single = timeEncoder(kappa, delta, b)
        spikes_single = tem_single.encode_precise(
            original.sinc_locs, original.sinc_amps, omega, t[-1], delta_t
        )
        rec_single = tem_single.decode(spikes_single, t, omega, delta_t)
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)
        assert (
            np.mean(((rec_single - y) ** 2)[start_index:end_index]) / np.mean(y ** 2)
            < 1e-3
        )


class TestTimeEncoderSingleSignalMultiChannel:
    def test_TEM_can_reconstruct_standard_encoding_ex1(self):
        kappa = [3, 3, 3, 3]
        delta = [1, 1, 1, 1]
        int_shift = [-1, -0.5, 0, 0.5]

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 20, delta_t)
        original = bandlimitedSignal(t, delta_t, omega, seed=10)
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1

        tem_mult = timeEncoder(kappa, delta, b, n_channels=4, integrator_init=int_shift)
        spikes_mult = tem_mult.encode(y, delta_t)
        rec_mult = tem_mult.decode(spikes_mult, t, omega, delta_t)
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)
        assert (
            np.mean(((rec_mult - y) ** 2)[start_index:end_index]) / np.mean(y ** 2)
            < 1e-3
        )

    def test_TEM_can_reconstruct_precise_encoding_ex1(self):
        kappa = [3, 3, 3, 3]
        delta = [1, 1, 1, 1]
        int_shift = [-1, -0.5, 0, 0.5]

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 20, delta_t)
        original = bandlimitedSignal(t, delta_t, omega, seed=10)
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1

        tem_mult = timeEncoder(kappa, delta, b, n_channels=4, integrator_init=int_shift)
        spikes_mult = tem_mult.encode_precise(
            original.sinc_locs, original.sinc_amps, omega, t[-1], delta_t
        )
        rec_mult = tem_mult.decode(spikes_mult, t, omega, delta_t)
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)
        assert (
            np.mean(((rec_mult - y) ** 2)[start_index:end_index]) / np.mean(y ** 2)
            < 1e-3
        )
