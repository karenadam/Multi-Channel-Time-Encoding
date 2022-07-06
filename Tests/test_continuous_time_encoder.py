import sys
import os
import numpy as np


sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/..")
import src.signals
from src import *


class TestContinuousTimeEncoder:
    def test_consistent_constraints_sos(self):
        kappa = 1
        delta = 1

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 15, delta_t)
        np.random.seed(10)
        original = src.signals.bandlimitedSignal(
            omega, sinc_locs=np.arange(0, 25, np.pi / omega)
        )
        signals = src.signals.bandlimitedSignals(omega)
        signals.add(original)
        b = 2

        tem_params = TEMParams(kappa, delta, b, mixing_matrix=[[1]])
        spikes_single = encoder.ContinuousEncoder(tem_params).encode(
            signals, signal_end_time=15
        )
        q = decoder.SSignalMChannelDecoder(
            tem_params, Omega=omega
        ).get_measurement_vector(spikes_single)
        q_gt = original.get_precise_integral(
            spikes_single[0][:-1], spikes_single[0][1:]
        )
        assert np.allclose(q, q_gt)

    def test_consistent_constraints_per(self):
        kappa = 1
        delta = 1

        period = 5
        delta_t = 1e-4
        t = np.arange(0, 15, delta_t)
        np.random.seed(10)
        n_components = 5
        original = src.signals.periodicBandlimitedSignal(
            period, n_components, np.random.random((2 * n_components - 1))
        )
        signals = src.signals.periodicBandlimitedSignals(period)
        signals.add(original)
        b = 2

        tem_params = TEMParams(kappa, delta, b, mixing_matrix=[[1]])
        spikes_single = encoder.ContinuousEncoder(tem_params).encode(
            signals, signal_end_time=15
        )
        q = decoder.SSignalMChannelDecoder(
            tem_params, periodic=True, period=period, n_components=n_components
        ).get_measurement_vector(spikes_single)
        q_gt = original.get_precise_integral(
            spikes_single[0][:-1], spikes_single[0][1:]
        )
        assert np.allclose(q, q_gt, atol=1e-5)
