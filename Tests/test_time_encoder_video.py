import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../")
from src import *


def get_f_s_coeffs_from_time_encoded_video(
    video, TEM_locations, rank=10, num_spikes=None, plot=False
):

    signals = Signal.periodicBandlimitedSignals(period=video.periods[-1])
    deltas = []
    if num_spikes is None:
        num_spikes = video.periods[-1] + 8.5

    for TEM_l in TEM_locations:
        signal_l = video.get_time_signal(TEM_l)
        signals.add(signal_l)
        deltas.append(
            signal_l.get_precise_integral(0, video.periods[-1]) / (2 * num_spikes + 0.5)
        )

    kappa, b = 1, 0
    tem_mult = TEMParams(kappa, deltas, b, np.eye(len(TEM_locations)))
    end_time = video.periods[-1]
    spikes = Encoder.ContinuousEncoder(tem_mult).encode(
        signals, end_time, tolerance=1e-14, with_start_time=False
    )

    decoder = Decoder.MSignalMChannelDecoder(
        tem_mult,
        periodic=True,
        period=video.periods[-1],
        n_components=video.num_components[-1],
    )
    integrals = decoder.get_measurement_vector(spikes)
    (
        integral_start_coordinates,
        integral_end_coordinates,
    ) = decoder.get_integral_start_end_coordinates(spikes, TEM_locations)
    coefficients = video.get_coefficients_from_integrals(
        integral_start_coordinates, integral_end_coordinates, integrals
    )

    return coefficients


class TestTimeEncoderVideo:
    def test_time_encode_video_min_space_sampling_odd_pixel_num(self):
        height = 5
        width = 5
        length = 10
        t_d_samples = np.random.random((height, width, length))
        opt = {"time_domain_samples": t_d_samples}
        video = MultiDimPeriodicSignal(opt)
        TEM_locations = [[v, h] for v in range(height) for h in range(width)]
        f_s_coefficients = get_f_s_coeffs_from_time_encoded_video(
            video, TEM_locations=TEM_locations, num_spikes=length + 1
        )
        assert np.allclose(f_s_coefficients, video.freq_domain_samples.flatten())

    def test_time_encode_video_oversampled_space_odd_pixel_num(self):
        height = 5
        width = 5
        length = 12
        t_d_samples = np.random.random((height, width, length))
        opt = {"time_domain_samples": t_d_samples}
        video = MultiDimPeriodicSignal(opt)
        TEM_locations = [
            [0.25 + 0.5 * v, 0.25 * 0.5 * h]
            for v in range(2 * height)
            for h in range(2 * width)
        ]
        f_s_coefficients = get_f_s_coeffs_from_time_encoded_video(
            video, TEM_locations=TEM_locations, num_spikes=length / 4 + 1
        )
        assert np.allclose(
            f_s_coefficients, video.freq_domain_samples.flatten(), rtol=1e-4
        )

    def test_time_encode_video_min_space_sampling_even_pixel_num(self):
        height = 4
        width = 4
        length = 10
        t_d_samples = np.random.random((height, width, length))
        opt = {"time_domain_samples": t_d_samples}
        video = MultiDimPeriodicSignal(opt)
        TEM_locations = [[v, h] for v in range(height) for h in range(width)]
        f_s_coefficients = get_f_s_coeffs_from_time_encoded_video(
            video, TEM_locations=TEM_locations, num_spikes=length + 1
        )
        assert np.allclose(f_s_coefficients, video.freq_domain_samples.flatten())
