import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/..")
from src import *


class TestTimeEncoderSingleSignalSingleChannel:
    def test_time_encoder_creator(self):
        TEM = TEMParams(kappa=1, delta=1, b=1, mixing_matrix=[[1]])
        assert TEM.kappa == [1], "kappa was falsely assigned"
        assert TEM.delta == [1], "delta was falsely assigned"
        assert TEM.b == [1], "b was falsely assigned"

        TEM3 = TEMParams(kappa=[1], delta=1, b=1, mixing_matrix=[[1], [1]])
        assert len(TEM3.kappa) == 2

    def test_can_reconstruct_standard_encoding(self):
        kappa = 1
        delta = 1

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 15, delta_t)
        np.random.seed(10)
        original = Signal.bandlimitedSignal(
            omega, sinc_locs=np.arange(0, 25, np.pi / omega)
        )
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1

        tem_params = TEMParams(kappa, delta, b, mixing_matrix=[[1]])
        spikes_single = Encoder.DiscreteEncoder(tem_params).encode(
            original, signal_end_time=15, delta_t=delta_t
        )
        rec_single = Decoder.SSignalMChannelDecoder(
            tem_params, periodic=False, Omega=omega
        ).decode(spikes_single, t)
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)

        assert (
            np.mean(((rec_single - y) ** 2)[start_index:end_index]) / np.mean(y**2)
            < 1e-3
        )

    def test_can_reconstruct_standard_encoding_2(self):
        kappa = 1
        delta = 1

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 15, delta_t)
        np.random.seed(10)
        original = Signal.bandlimitedSignal(
            omega, sinc_locs=np.arange(0, 25, np.pi / omega)
        )
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1

        tem_params = TEMParams(kappa, delta, b, mixing_matrix=[[1]])

        encoder = Encoder.DiscreteEncoder(tem_params)
        spikes_single = encoder.encode(original, signal_end_time=15, delta_t=delta_t)
        decoder = Decoder.SSignalMChannelDecoder(
            tem_params, periodic=False, Omega=omega
        )
        rec_single = decoder.decode(
            spikes_single,
            t,
        )
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)

        assert (
            np.mean(((rec_single - y) ** 2)[start_index:end_index]) / np.mean(y**2)
            < 1e-3
        )

    def test_can_reconstruct_precise_encoding(self):
        kappa = 1
        delta = 1

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 15, delta_t)
        np.random.seed(10)
        x_param = []
        original = Signal.bandlimitedSignal(
            omega, sinc_locs=np.arange(0, 15, np.pi / omega)
        )
        x_param.append(original)
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1
        signal = SignalCollection.bandlimitedSignals(omega)
        signal.add(original)

        tem_params = TEMParams(kappa, delta, b, mixing_matrix=[[1]])
        spikes_single = Encoder.ContinuousEncoder(tem_params).encode(signal, t[-1])
        rec_single = Decoder.SSignalMChannelDecoder(
            tem_params, periodic=False, Omega=omega
        ).decode(
            spikes_single,
            t,
        )
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)
        assert (
            np.mean(((rec_single - y) ** 2)[start_index:end_index]) / np.mean(y**2)
            < 1e-3
        )
