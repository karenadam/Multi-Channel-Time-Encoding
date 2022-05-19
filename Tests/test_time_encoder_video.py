import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../")
from src import *


def get_f_s_coeffs_from_time_encoded_video(video, TEM_locations, rank = 10, num_spikes = None, plot = False):

    signals = Signal.periodicBandlimitedSignals(period = video.periods[-1])
    deltas = []
    if num_spikes is None:
        num_spikes = video.periods[-1 ] +8.5

    for TEM_l in TEM_locations:
        signal_l = video.get_time_signal(TEM_l)
        signals.add(signal_l)
        deltas.append(signal_l.get_precise_integral(0 ,video.periods[-1] ) /( 2 *num_spikes))

    kappa, b = 1, 0
    tem_mult = TEMParams(kappa, deltas, b, np.eye(len(TEM_locations)))
    end_time = video.periods[-1]
    spikes = Encoder.ContinuousEncoder(tem_mult).encode(signals, end_time, tol = 1e-14, with_start_time = False)

    integrals, integral_start_coordinates, integral_end_coordinates = Decoder.MSignalMChannelDecoder \
        (tem_mult).get_vid_constraints(spikes, TEM_locations)
    coefficients = video.get_coefficients_from_integrals(integral_start_coordinates, integral_end_coordinates, integrals)

    return coefficients


class TestTimeEncoderVideo:
    def test_time_encode_video_min_space_sampling(self):
        height =5
        width = 5
        length = 10
        t_d_samples = np.random.random((height, width, length))
        opt = {'time_domain_samples': t_d_samples}
        video = MultiDimPeriodicSignal(opt)
        TEM_locations= [[v, h] for v in range(height) for h in range(width)]
        f_s_coefficients = get_f_s_coeffs_from_time_encoded_video(video, TEM_locations= TEM_locations , num_spikes = 2*length+1)
        assert np.allclose(f_s_coefficients, video.freq_domain_samples.flatten())
        
