import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../src")
from src import *


class TestTimeEncoderMultiSignalMultiChannel:
    def test_can_reconstruct_standard_encoding_with_2_by_2_mixing_one_shot(self):
        kappa = [1, 1]
        delta = [2, 1]
        b = [5, 3]
        int_shift = [-1, -0.1]

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 25, delta_t)
        np.random.seed(10)
        original1 = Signal.bandlimitedSignal(omega)
        original1.random(t)
        np.random.seed(11)
        original2 = Signal.bandlimitedSignal(omega)
        original2.random(t)
        signals = Signal.bandlimitedSignals(omega, sinc_locs=[], sinc_amps=[])
        signals.add(original1)
        signals.add(original2)
        y = signals.sample(t)
        y = np.atleast_2d(y)
        A = [[0.9, 0.1], [0.2, 0.8]]

        tem_params = TEMParams(kappa, delta, b, A, integrator_init=int_shift)
        spikes_mult = Encoder.ContinuousEncoder(tem_params).encode(signals, t[-1])

        rec_mult = Decoder.MSignalMChannelDecoder(
            tem_params, periodic=False, sinc_locs=original1.get_sinc_locs(), Omega=omega
        ).decode(
            spikes_mult,
            t,
        )

        start_index = int(y.shape[1] / 10)
        end_index = int(y.shape[1] * 9 / 10)

        assert (
            np.mean(((rec_mult[0, :] - y[0, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )
        assert (
            np.mean(((rec_mult[1, :] - y[1, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )

    def test_can_reconstruct_precise_encoding_with_3_by_2_mixing_one_shot(self):
        kappa = [1, 1, 1]
        delta = [2, 1, 1]
        b = [2, 2, 2]
        int_shift = [-1, -0.1, 0.2]

        omega = np.pi
        delta_t = 1e-4
        end_time = 25
        t = np.arange(0, 25, delta_t)
        y_param = []
        np.random.seed(10)
        original1 = Signal.bandlimitedSignal(omega)
        original1.random(t)
        np.random.seed(11)
        original2 = Signal.bandlimitedSignal(omega)
        original2.random(t)
        y = np.zeros((2, len(t)))
        y[0, :] = original1.sample(t)
        y[1, :] = original2.sample(t)
        y = np.atleast_2d(y)
        A = [[0.9, 0.1], [0.2, 0.8], [1, 1]]
        y_param = Signal.bandlimitedSignals(omega, sinc_locs=[], sinc_amps=[])
        y_param.add(original1)
        y_param.add(original2)

        tem_params = TEMParams(kappa, delta, b, A, integrator_init=int_shift)
        spikes_mult = Encoder.ContinuousEncoder(tem_params).encode(y_param, end_time)

        rec_mult = Decoder.MSignalMChannelDecoder(
            tem_params, periodic=False, sinc_locs=original1.get_sinc_locs(), Omega=omega
        ).decode(
            spikes_mult,
            t,
        )

        start_index = int(y.shape[1] / 10)
        end_index = int(y.shape[1] * 9 / 10)

        assert (
            np.mean(((rec_mult[0, :] - y[0, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )
        assert (
            np.mean(((rec_mult[1, :] - y[1, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )

    def test_can_reconstruct_precise_encoding_with_3_by_2_mixing_one_shot_no_fixed_sinc_locs(
        self,
    ):
        kappa = [1, 1, 1]
        delta = [2, 1, 1]
        b = [2, 2, 2]
        int_shift = [-1, -0.1, 0.2]

        omega = np.pi
        delta_t = 1e-4
        end_time = 25
        t = np.arange(0, 25, delta_t)
        y_param = []
        np.random.seed(10)
        original1 = Signal.bandlimitedSignal(omega)
        original1.random(t)
        np.random.seed(11)
        original2 = Signal.bandlimitedSignal(omega)
        original2.random(t)
        y = np.zeros((2, len(t)))
        y[0, :] = original1.sample(t)
        y[1, :] = original2.sample(t)
        y = np.atleast_2d(y)
        A = [[0.9, 0.1], [0.2, 0.8], [1, 1]]
        y_param = Signal.bandlimitedSignals(omega, sinc_locs=[], sinc_amps=[])
        y_param.add(original1)
        y_param.add(original2)

        # signal_set = bandlimitedSignals(omega)
        # signal_set.add(original1)
        # signal_set.add(original2)

        tem_params = TEMParams(kappa, delta, b, A, integrator_init=int_shift)
        spikes_mult = Encoder.ContinuousEncoder(tem_params).encode(y_param, end_time)

        rec_mult = Decoder.MSignalMChannelDecoder(
            tem_params, periodic=False, sinc_locs=original1.get_sinc_locs(), Omega=omega
        ).decode(
            spikes_mult,
            t,
        )

        start_index = int(y.shape[1] / 10)
        end_index = int(y.shape[1] * 9 / 10)

        assert (
            np.mean(((rec_mult[0, :] - y[0, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )
        assert (
            np.mean(((rec_mult[1, :] - y[1, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )

    def test_params_TEM_correct(self):
        TEM = TEMParams(kappa=1, delta=1, b=1, mixing_matrix=[[1]] * 2)
        assert TEM.kappa == [1, 1], "kappa was falsely assigned"
        assert TEM.delta == [1, 1], "delta was falsely assigned"
        assert TEM.b == [1, 1], "b was falsely assigned"
