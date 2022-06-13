import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/..")
from src import *


class TestTimeEncoderSingleSignalMultiChannel:
    def test_tem_params_repr(self):
        kappa = [4, 3.2, 2.8, 3]
        delta = [0.8, 1.2, 0.6, 1]
        b = 1
        int_shift = max(delta) / 2

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 20, delta_t)
        np.random.seed(10)
        original = Signal.bandlimitedSignal(
            omega, sinc_locs=np.arange(0, 20, np.pi / omega)
        )
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1

        tem_mult = TEMParams(
            kappa, delta, b, mixing_matrix=[[1]] * 4, integrator_init=[int_shift]
        )
        spikes_mult = Encoder.DiscreteEncoder(tem_mult).encode(
            original, signal_end_time=20, delta_t=delta_t
        )
        print(tem_mult)
        print(spikes_mult)
        print(spikes_mult.__repr__())

    def test_tem_params_throws_error_when_mismatch_in_params(self):
        kappa = [4, 3.2, 2.8, 3]
        delta = [0.8, 1.2, 0.6]
        b = 1
        int_shift = max(delta) / 2

        try:
            tem_mult = TEMParams(
                kappa, delta, b, mixing_matrix=[[1]] * 4, integrator_init=[int_shift]
            )
        except:
            return
        assert False

    def test_can_reconstruct_standard_encoding_ex1(self):
        kappa = [4, 3.2, 2.8, 3]
        delta = [0.8, 1.2, 0.6, 1]
        int_shift = max(delta) / 2

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 20, delta_t)
        np.random.seed(10)
        original = Signal.bandlimitedSignal(
            omega, sinc_locs=np.arange(0, 20, np.pi / omega)
        )
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1

        tem_mult = TEMParams(
            kappa, delta, b, mixing_matrix=[[1]] * 4, integrator_init=[int_shift]
        )
        spikes_mult = Encoder.DiscreteEncoder(tem_mult).encode(
            original, signal_end_time=20, delta_t=delta_t
        )
        rec_mult = Decoder.SSignalMChannelDecoder(
            tem_mult, periodic=False, Omega=omega
        ).decode(spikes_mult, t)
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)
        assert (
            np.mean(((rec_mult - y) ** 2)[start_index:end_index]) / np.mean(y**2)
            < 1e-3
        )

    def test_TEM_can_reconstruct_standard_encoding_ex2(self):
        kappa = [1, 1, 1, 1]
        delta = [1, 1, 1, 1]
        int_shift = [-1, -0.5, 0, 0.5]

        omega = 2 * np.pi
        delta_t = 1e-4
        t = np.arange(0, 20, delta_t)
        np.random.seed(10)
        original = Signal.bandlimitedSignal(
            omega, sinc_locs=np.arange(0, 20, np.pi / omega)
        )
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1.1

        tem_mult = TEMParams(
            kappa, delta, b, mixing_matrix=[[1]] * 4, integrator_init=int_shift
        )
        spikes_mult = Encoder.DiscreteEncoder(tem_mult).encode(
            original, signal_end_time=20, delta_t=delta_t
        )
        print(spikes_mult)
        rec_mult = Decoder.SSignalMChannelDecoder(
            tem_mult, periodic=False, Omega=omega
        ).decode(
            spikes_mult,
            t,
        )
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)
        assert (
            np.mean(((rec_mult - y) ** 2)[start_index:end_index]) / np.mean(y**2)
            < 2e-3
        )

    # def test_can_reconstruct_standard_encoding_with_recursive_mixing_alg(self):
    #     kappa = [1]
    #     delta = [0.5]
    #     b = 1.5

    #     omega = np.pi
    #     delta_t = 1e-4
    #     t = np.arange(0, 25, delta_t)
    #     np.random.seed(10)
    #     original = bandlimitedSignal(omega)
    #     original.random(t, padding=2)
    #     y = original.sample(t)
    #     y = np.atleast_2d(y)
    #     A = [[1]]

    #     tem_mult = TEMParams(kappa, delta, b, A)
    #     spikes_mult = DiscreteEncoder(tem_mult).encode(y, delta_t)
    #     rec_mult = SSignalMChannelDecoder(tem_mult).decode_recursive(
    #         spikes_mult, t, original.get_sinc_locs(), omega, delta_t, num_iterations=100
    #     )

    #     start_index = int(y.shape[1] / 10)
    #     end_index = int(y.shape[1] * 9 / 10)

    #     assert (
    #         np.mean(((rec_mult[0, :] - y[0, :]) ** 2)[start_index:end_index])
    #         / np.mean(y[0, :] ** 2)
    #         < 1e-3
    #     )

    def test_can_reconstruct_standard_encoding_with_one_shot_mixing_alg(self):
        kappa = [0.5]
        delta = [0.1]
        b = 1.5

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 25, delta_t)
        np.random.seed(10)
        original = Signal.bandlimitedSignal(
            omega, sinc_locs=np.arange(2, 23, np.pi / omega)
        )
        signal = SignalCollection.bandlimitedSignals(np.pi, sinc_locs=[], sinc_amps=[])
        signal.add(original)
        y = original.sample(t)
        y = np.atleast_2d(y)
        A = [[1]]

        tem_mult = TEMParams(kappa, delta, b, A)
        spikes_mult = Encoder.ContinuousEncoder(tem_mult).encode(signal, t[-1])
        rec_mult = Decoder.MSignalMChannelDecoder(
            tem_mult, periodic=False, sinc_locs=original.get_sinc_locs(), Omega=omega
        ).decode(
            spikes_mult,
            t,
        )

        start_index = int(y.shape[1] / 10)
        end_index = int(y.shape[1] * 9 / 10)

        print(rec_mult[0, :])
        print(y[0, :])

        assert (
            np.mean(((rec_mult[0, :] - y[0, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )

    def test_can_reconstruct_standard_encoding_with_one_shot_mixing_alg3(self):
        kappa = [50]
        delta = [0.001]
        b = 1.5

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 25, delta_t)
        np.random.seed(10)
        original = Signal.bandlimitedSignal(
            omega, sinc_locs=np.arange(2, 23, np.pi / omega)
        )
        signal = SignalCollection.bandlimitedSignals(np.pi, sinc_locs=[], sinc_amps=[])
        signal.add(original)
        y = original.sample(t)
        y = np.atleast_2d(y)
        A = [[1]]

        tem_mult = TEMParams(kappa, delta, b, A)
        spikes_mult = Encoder.ContinuousEncoder(tem_mult).encode(signal, t[-1])
        rec_mult = Decoder.MSignalMChannelDecoder(
            tem_mult, periodic=False, sinc_locs=original.get_sinc_locs(), Omega=omega
        ).decode(
            spikes_mult,
            t,
        )

        start_index = int(y.shape[1] / 10)
        end_index = int(y.shape[1] * 9 / 10)

        print(rec_mult[0, :])
        print(y[0, :])

        assert (
            np.mean(((rec_mult[0, :] - y[0, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )

    def test_can_reconstruct_standard_encoding_with_one_shot_mixing_alg2(self):
        kappa = [1, 1]
        delta = [2, 1]
        b = 1.5
        int_shift = [-1, -0.1]

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 25, delta_t)
        np.random.seed(10)
        original = Signal.bandlimitedSignal(
            omega, sinc_locs=np.arange(2, 23, np.pi / omega)
        )
        signal = SignalCollection.bandlimitedSignals(np.pi, sinc_locs=[], sinc_amps=[])
        signal.add(original)
        y = original.sample(t)
        y = np.atleast_2d(y)
        A = [[1], [2]]

        tem_mult = TEMParams(kappa, delta, b, A, integrator_init=int_shift)
        spikes_mult = Encoder.ContinuousEncoder(tem_mult).encode(signal, t[-1])
        rec_mult = Decoder.MSignalMChannelDecoder(
            tem_mult, periodic=False, sinc_locs=original.get_sinc_locs(), Omega=omega
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
