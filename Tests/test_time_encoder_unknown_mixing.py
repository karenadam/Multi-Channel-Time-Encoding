import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/..")
from src import *


class TestTimeEncoderMultiSignalMultiChannel:
    def test_can_reconstruct_standard_encoding_with_4_by_2_mixing_sinc(self):
        kappa = [1, 1, 1, 1]
        delta = [2, 1, 1, 1]
        b = [5, 3, 4, 1]
        int_shift = [-1, -0.1, 0, -0.5]

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 25, delta_t)
        np.random.seed(10)
        original1 = src.signals.bandlimitedSignal(
            omega, sinc_locs=np.arange(0, 25, np.pi / omega)
        )
        # original1.random(t)
        np.random.seed(11)
        original2 = src.signals.bandlimitedSignal(
            omega, sinc_locs=np.arange(0, 25, np.pi / omega)
        )
        # original2.random(t)
        original = src.signals.bandlimitedSignals(
            omega, sinc_locs=np.arange(0, 25, np.pi / omega)
        )
        original.add(original1)
        original.add(original2)
        y = np.zeros((2, len(t)))
        y[0, :] = original1.sample(t)
        y[1, :] = original2.sample(t)
        y = np.atleast_2d(y)
        A = [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.5, 0.5]]

        tem_params = TEMParams(kappa, delta, b, A, integrator_init=int_shift)
        spikes_mult = encoder.DiscreteEncoder(tem_params).encode(
            original, signal_end_time=25, delta_t=delta_t
        )
        rec_mult = decoder.UnknownMixingDecoder(
            tem_params, sinc_locs=original1.get_sinc_locs(), Omega=omega
        ).decode(
            spikes_mult,
            2,
            t,
        )

        start_index = int(y.shape[1] / 10)
        end_index = int(y.shape[1] * 9 / 10)

        assert (
            np.mean(
                ((rec_mult[0, :] - np.array(A).dot(y)[0, :]) ** 2)[
                    start_index:end_index
                ]
            )
            / np.mean(np.array(A).dot(y)[0, :] ** 2)
            < 1e-3
        )
        assert (
            np.mean(
                ((rec_mult[1, :] - np.array(A).dot(y)[1, :]) ** 2)[
                    start_index:end_index
                ]
            )
            / np.mean(np.array(A).dot(y)[0, :] ** 2)
            < 1e-3
        )

    def test_can_reconstruct_standard_encoding_with_4_by_2_mixing_periodic(self):
        kappa = [0.7] * 4
        delta = [2, 1, 1, 1]
        b = [5, 3, 4, 1]
        int_shift = [-1, -0.1, 0, -0.5]

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 25, delta_t)
        np.random.seed(10)
        period = 20
        n_components = 15
        original1 = src.signals.periodicBandlimitedSignal(
            period, n_components, np.random.random(size=(n_components)).tolist()
        )
        np.random.seed(11)
        original2 = src.signals.periodicBandlimitedSignal(
            period, n_components, np.random.random(size=(n_components)).tolist()
        )
        original = src.signals.periodicBandlimitedSignals(period)
        original.add(original1)
        original.add(original2)
        y = np.zeros((2, len(t)), dtype="complex")
        y[0, :] = original1.sample(t)
        y[1, :] = original2.sample(t)
        y = np.atleast_2d(y)
        A = [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.5, 0.5]]

        tem_params = TEMParams(kappa, delta, b, A, integrator_init=int_shift)
        spikes_mult = encoder.ContinuousEncoder(tem_params).encode(
            original, signal_end_time=25, tolerance = delta_t
        )
        rec_mult = decoder.UnknownMixingDecoder(
            tem_params, periodic=True, n_components=n_components, period=period
        ).decode(spikes_mult, 2, t)

        start_index = int(y.shape[1] / 10)
        end_index = int(y.shape[1] * 9 / 10)

        assert (
            np.mean(
                ((rec_mult[0, :] - np.array(A).dot(y)[0, :]) ** 2)[
                    start_index:end_index
                ]
            )
            / np.mean(np.array(A).dot(y)[0, :] ** 2)
            < 1e-3
        )
        assert (
            np.mean(
                ((rec_mult[1, :] - np.array(A).dot(y)[1, :]) ** 2)[
                    start_index:end_index
                ]
            )
            / np.mean(np.array(A).dot(y)[0, :] ** 2)
            < 1e-3
        )
