import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../Source")
from Time_Encoder import timeEncoder
from Signal import *

import scipy
import scipy.signal


class TestBandlimitedSignal():
    def test_integral(self):
        delta_t = 0.1
        t = np.arange(0,10,delta_t)
        sinc_locs = [1.3,2.8,7.2]
        sinc_amps = [1,5,3]
        Omega = np.pi
        signal = bandlimitedSignal(Omega, sinc_locs, sinc_amps)
        t_int_0 = np.arange(2,7,0.01)
        t_int_1 = np.arange(4,8,0.01)
        discrete_integral_0 = signal.get_total_integral(t_int_0)
        discrete_integral_1 = signal.get_total_integral(t_int_1)
        precise_integral = signal.get_precise_integral([2,4],[7,8])

        assert (np.abs(discrete_integral_0 - precise_integral[0])<np.abs(discrete_integral_0)*1e-2)
        assert (np.abs(discrete_integral_1 - precise_integral[1])<np.abs(discrete_integral_1)*1e-2)

class TestPiecewiseConstantSignal():
    def test_sampling(self):
        discontinuities = [1,2]
        values = [1]
        signal = piecewiseConstantSignal(discontinuities, values)
        samples = signal.sample([0,0.1,1.2,1.5, 1.7,2.3, 3.5])
        assert(samples[0]==0)
        assert(samples[1]==0)
        assert(samples[2]==1)
        assert(samples[3]==1)
        assert(samples[4]==1)
        assert(samples[5]==0)
        assert(samples[6]==0)


    def test_sampling2(self):
        discontinuities = [1,2,5,6]
        values = [1,4,-2]
        signal = piecewiseConstantSignal(discontinuities, values)
        samples = signal.sample([0,0.1,1.2,1.5, 1.7,2.3, 3.5, 5.6, 6.5])
        assert(samples[0]==0)
        assert(samples[1]==0)
        assert(samples[2]==1)
        assert(samples[3]==1)
        assert(samples[4]==1)
        assert(samples[5]==4)
        assert(samples[6]==4)
        assert(samples[7]==-2)
        assert(samples[8]==0)


class TestLPFPCSsignal():
    def test_sampling(self):
        omega = np.pi
        discontinuities = [1,2,5,6]
        values = [1,4,-2]
        t = np.arange(0,10,0.1)
        signal = piecewiseConstantSignal(discontinuities, values)
        samples = signal.sample(t)
        filtered_signal = signal.low_pass_filter(omega)
        samples_filtered = filtered_signal.sample(t)

        sampled_sinc = sinc(t-5, omega)
        samples_discrete_filtered = scipy.signal.convolve(samples, sampled_sinc)*(t[1]-t[0])
        offset = int(len(sampled_sinc)/2)

        assert(np.abs(samples_filtered[40] - samples_discrete_filtered[offset+40])<1e-2)





