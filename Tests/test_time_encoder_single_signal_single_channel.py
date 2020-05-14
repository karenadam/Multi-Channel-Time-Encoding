import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../Source")
from Time_Encoder import timeEncoder
from Signal import (
    bandlimitedSignal,
    bandlimitedSignals,
    periodicBandlimitedSignal,
    periodicBandlimitedSignals,
)
from Spike_Times import spikeTimes


class TestTimeEncoderSingleSignalSingleChannel:
    def test_time_encoder_creator(self):
        TEM = timeEncoder(kappa=1, delta=1, b=1, mixing_matrix=[[1]])
        assert TEM.kappa == [1], "kappa was falsely assigned"
        assert TEM.delta == [1], "delta was falsely assigned"
        assert TEM.b == [1], "b was falsely assigned"

        TEM2 = timeEncoder(kappa=[1], delta=1, b=1, mixing_matrix=[[1], [1]])
        assert len(TEM2.kappa) == 2

    def test_can_reconstruct_standard_encoding(self):
        kappa = 1
        delta = 1

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 15, delta_t)
        np.random.seed(10)
        original = bandlimitedSignal(omega)
        original.random(t)
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1

        tem_single = timeEncoder(kappa, delta, b, mixing_matrix=[[1]])
        spikes_single = tem_single.encode(y, delta_t)
        rec_single = tem_single.decode(spikes_single, t, periodic=False, Omega=omega)
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
        np.random.seed(10)
        x_param = []
        original = bandlimitedSignal(omega)
        original.random(t)
        x_param.append(original)
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1
        signal = bandlimitedSignals(omega)
        signal.add(original)

        tem_single = timeEncoder(kappa, delta, b, mixing_matrix=[[1]])
        spikes_single = tem_single.encode_precise(signal, t[-1])
        rec_single = tem_single.decode(spikes_single, t, periodic=False, Omega=omega)
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)
        assert (
            np.mean(((rec_single - y) ** 2)[start_index:end_index]) / np.mean(y ** 2)
            < 1e-3
        )
