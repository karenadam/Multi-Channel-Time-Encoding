import sys
import os
import numpy as np

# sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../src")
sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/..")

from src import *

import scipy
import scipy.signal


def normed_difference(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


class TestBandlimitedSignal:
    def test_integral(self):
        delta_t = 0.1
        t = np.arange(0, 10, delta_t)
        sinc_locs = [1.3, 2.8, 7.2]
        sinc_amps = [1, 5, 3]
        Omega = np.pi
        signal = Signal.bandlimitedSignal(Omega, sinc_locs, sinc_amps)
        t_int_0 = np.arange(2, 7, 0.01)
        t_int_1 = np.arange(4, 8, 0.01)
        discrete_integral_0 = np.sum(signal.sample(t_int_0)) * 0.01
        discrete_integral_1 = np.sum(signal.sample(t_int_1)) * 0.01
        precise_integral = signal.get_precise_integral([2, 4], [7, 8])

        assert (
            np.abs(discrete_integral_0 - precise_integral[0])
            < np.abs(discrete_integral_0) * 1e-2
        )
        assert (
            np.abs(discrete_integral_1 - precise_integral[1])
            < np.abs(discrete_integral_1) * 1e-2
        )


class TestBandlimitedSignals:
    def test_mixing(self):
        sinc_locs = [1, 2, 3]
        sinc_amps = [[1, 0, 1], [1, 1, 0]]
        omega = np.pi
        signals = SignalCollection.bandlimitedSignals(omega, sinc_locs, sinc_amps)
        mixed_amplitudes = signals.mix_amplitudes([[2, 1], [1, 0]])
        expected_amplitudes = [3, 1, 2, 1, 0, 1]

        assert np.linalg.norm(mixed_amplitudes - expected_amplitudes) < 1e-6


class TestPiecewiseConstantSignal:
    def test_sampling(self):
        discontinuities = [1, 2]
        values = [1]
        signal = Signal.piecewiseConstantSignal(discontinuities, values)
        samples = signal.sample([0, 0.1, 1.2, 1.5, 1.7, 2.3, 3.5])
        assert samples[0] == 0
        assert samples[1] == 0
        assert samples[2] == 1
        assert samples[3] == 1
        assert samples[4] == 1
        assert samples[5] == 0
        assert samples[6] == 0

    def test_sampling2(self):
        discontinuities = [1, 2, 5, 6]
        values = [1, 4, -2]
        signal = Signal.piecewiseConstantSignal(discontinuities, values)
        samples = signal.sample([0, 0.1, 1.2, 1.5, 1.7, 2.3, 3.5, 5.6, 6.5])
        assert samples[0] == 0
        assert samples[1] == 0
        assert samples[2] == 1
        assert samples[3] == 1
        assert samples[4] == 1
        assert samples[5] == 4
        assert samples[6] == 4
        assert samples[7] == -2
        assert samples[8] == 0


class TestPiecewiseConstantSignals:
    def test_creation(self):
        discontinuities = [[1, 2], [1.5, 4]]
        values = [[1], [-2]]
        signals = SignalCollection.piecewiseConstantSignals(discontinuities, values)
        signal_1 = signals[1]
        assert signal_1.discontinuities == [1.5, 4]
        assert signal_1.values == [-2]

    def test_sampling(self):
        discontinuities = [[1, 2, 4, 6, 7, 8], [1.5, 4, 7, 9, 10]]
        values = [[1, 3, 4, -2, 3], [-2, 1, 3, -1]]
        signals = SignalCollection.piecewiseConstantSignals(discontinuities, values)
        signal_1 = signals[1]

        sample_locs = [0, 1.5, 3]
        samples_1_matrix_approach = signals.sample(sample_locs, np.pi)[3:].T
        samples_1_loop_approach = signal_1.low_pass_filter(np.pi).sample(
            np.array(sample_locs)
        )

        assert (
            np.linalg.norm(samples_1_matrix_approach - samples_1_loop_approach) < 1e-6
        )


class TestLPFPCSsignal:
    def test_sampling(self):
        omega = np.pi
        discontinuities = [1, 2, 5, 6]
        values = [1, 4, -2]
        t = np.arange(0, 10, 0.1)
        signal = Signal.piecewiseConstantSignal(discontinuities, values)
        samples = signal.sample(t)
        filtered_signal = signal.low_pass_filter(omega)
        samples_filtered = filtered_signal.sample(t)

        sampled_sinc = Helpers.sinc(t - 5, omega)
        samples_discrete_filtered = scipy.signal.convolve(samples, sampled_sinc) * (
            t[1] - t[0]
        )
        offset = int(len(sampled_sinc) / 2)

        assert (
            np.abs(samples_filtered[40] - samples_discrete_filtered[offset + 40]) < 1e-2
        )


class TestBandlimitedPeriodicSignals:
    def test_signal_generation(self):
        omega = np.pi
        try:
            signal = Signal.periodicBandlimitedSignal(1 / omega, 3, [1, 2])
        except:
            return
        assert False

    def test_signal_generation_2(self):
        period = 1
        signal = Signal.periodicBandlimitedSignal(period, 3, [1, -1, 3 - 1j])
        t = np.arange(0, 1, 1e-3)
        samples = signal.sample(t)
        target = (
            1
            - 2 * np.cos(2 * np.pi * t)
            + 2 * 3 * np.cos(4 * np.pi * t)
            + 2 * np.sin(4 * np.pi * t)
        )
        assert np.linalg.norm(samples - target) < 1e-6

    def test_signal_sampling(self):
        period = 3
        signal = Signal.periodicBandlimitedSignal(period, 2, [1, 2])
        time = np.arange(0, 6, 0.5)
        samples = signal.sample(time)
        assert len(samples) == len(time)


class TestMultiDimPeriodicSignal:
    def test_signal_generation(self):
        opt = {"time_domain_samples": np.random.random((4, 4, 4))}
        signal = MultiDimPeriodicSignal(opt)

        opt = {"freq_domain_samples": np.random.random((4, 4, 4))}
        signal = MultiDimPeriodicSignal(opt)
        assert True

    def test_signal_sampling(self):
        opt = {"time_domain_samples": np.random.random((7, 5, 7))}
        signal = MultiDimPeriodicSignal(opt)
        coordinates = [0, 0, 0]
        assert (
            normed_difference(
                signal.sample(coordinates),
                opt["time_domain_samples"][
                    coordinates[0], coordinates[1], coordinates[2]
                ],
            )
            < 1e-6
        )

        coordinates = [2, 3, 1]
        assert (
            normed_difference(
                signal.sample(coordinates),
                opt["time_domain_samples"][
                    coordinates[0], coordinates[1], coordinates[2]
                ],
            )
            < 1e-6
        )

    def test_signal_sampling_nonint1(self):
        opt = {"time_domain_samples": np.random.random((3, 3, 1))}
        signal = MultiDimPeriodicSignal(opt)
        xcord, ycord = np.random.randint(0, 3), np.random.randint(0, 3)
        assert (
            normed_difference(
                signal.sample([xcord, ycord, 0.33]),
                opt["time_domain_samples"][xcord, ycord, 0],
            )
            < 1e-6
        )

    def test_signal_sampling_nonint1_2(self):
        opt = {"time_domain_samples": np.random.random((2, 4, 1))}
        signal = MultiDimPeriodicSignal(opt)
        xcord, ycord = np.random.randint(0, 2), np.random.randint(0, 2)
        assert (
            normed_difference(
                signal.sample([xcord, ycord, 0.33]),
                opt["time_domain_samples"][xcord, ycord, 0],
            )
            < 1e-6
        )

    def test_signal_sampling_int1_1(self):
        opt = {"time_domain_samples": np.random.random((3, 3, 3))}
        signal = MultiDimPeriodicSignal(opt)
        xcord, ycord = np.random.randint(0, 3), np.random.randint(0, 3)
        assert (
            normed_difference(
                signal.sample([xcord, ycord, 1]),
                opt["time_domain_samples"][xcord, ycord, 1],
            )
            < 1e-6
        )

    def test_signal_sampling_int1_2(self):
        opt = {"time_domain_samples": np.random.random((3, 3, 3))}
        signal = MultiDimPeriodicSignal(opt)
        xcord, tcord = np.random.randint(0, 3), np.random.randint(0, 3)
        assert (
            normed_difference(
                signal.sample([xcord, 0, tcord]),
                opt["time_domain_samples"][xcord, 0, tcord],
            )
            < 1e-6
        )

    def test_signal_sampling_nonint3(self):
        FT = np.zeros((3, 3, 3))
        FT[1, 1, 1] = 1

        def sampled(xcord, ycord, tcord):
            return (1 / 3**3) * np.cos(2 * np.pi / 3 * (xcord + ycord + tcord))

        opt = {"freq_domain_samples": FT}
        signal = MultiDimPeriodicSignal(opt)
        assert normed_difference(signal.sample([1, 2, 1]), sampled(1, 2, 1)) < 1e-6
        assert normed_difference(signal.sample([1, 1, 0.1]), sampled(1, 1, 0.1)) < 1e-6
        assert normed_difference(signal.sample([0, 2.1, 1]), sampled(0, 2.1, 1)) < 1e-6
        assert (
            normed_difference(signal.sample([1.5, 2.1, 1]), sampled(1.5, 2.1, 1)) < 1e-6
        )

        time_sig = signal.get_time_signal([1.5, 2.1])
        assert (
            normed_difference(np.real(time_sig.sample(1)), sampled(1.5, 2.1, 1)) < 1e-6
        )

    def test_signal_sampling_nonint4(self):
        FT = np.zeros((4, 4, 4))
        FT[3, 3, 3] = 1

        def sampled(xcord, ycord, tcord):
            return np.cos(2 * np.pi / 4 * (xcord + ycord + tcord)) + np.cos(
                np.pi / 4 * ycord
            )

        xcord_range = np.arange(0, 4, 1)
        ycord_range = np.arange(0, 4, 1)
        tcord_range = np.arange(0, 4, 1)
        TD_samples = np.zeros((4, 4, 4))
        for x in xcord_range:
            for y in ycord_range:
                for t in tcord_range:
                    TD_samples[x, y, t] = sampled(x, y, t)

        opt = {"time_domain_samples": TD_samples}

        signal = MultiDimPeriodicSignal(opt)
        time_sig = signal.get_time_signal([1, 2])
        print("BLA")
        print(time_sig.sample(0.8))
        print(signal.sample([1, 2, 0.8]))

        assert (
            signal.sample([2, 3, 1])
            <= signal.sample([2, 3, 0.5])
            <= signal.sample([2, 3, 0])
        )

    def test_signal_integration(self):
        opt = {"time_domain_samples": np.random.random((6, 5, 6))}
        TD = np.random.randint(0, 20, (5, 3, 6)) / 10
        # TD = np.ones((4,4,4))
        # TD[1,1,1] = 0

        opt = {"time_domain_samples": TD}

        signal = MultiDimPeriodicSignal(opt)
        x, y, target_time = [2.8, 1.2, 4.5]
        precise_integral = signal.get_precise_integral([x, y, target_time])
        BLsig_precise_integral = signal.get_time_signal([x, y]).get_precise_integral(
            0, target_time
        )
        delta_t = 1e-3
        estimated_integral = 0
        for t in np.arange(0, target_time, delta_t):
            estimated_integral += delta_t * signal.sample([x, y, t])
        print("integral = " + str(precise_integral))
        print(estimated_integral)
        assert normed_difference(precise_integral, estimated_integral) < 1e-3
        assert normed_difference(precise_integral, BLsig_precise_integral) < 1e-3

    def test_time_signal_object_creation(self):
        TD = np.random.randint(0, 20, (4, 5, 5)) / 10.0
        opt = {"time_domain_samples": TD}

        signal = MultiDimPeriodicSignal(opt)
        time_signal = signal.get_time_signal([2, 3])
        # import pudb
        # pudb.set_trace()
        estimated_value = time_signal.sample(0.1)
        target_value = signal.sample([2, 3, 0.1])
        assert normed_difference(estimated_value, target_value) < 1e-6

    def test_signal_rec_same_odd_dim(self):

        video = self.get_random_video(5, 5, 5)
        self.check_fourier_symmetry(video)
        self.check_reconstruction(video)
        video = self.get_random_video(3, 3, 5)
        self.check_fourier_symmetry(video)
        self.check_reconstruction(video)
        video = self.get_random_video(3, 3, 2)
        self.check_fourier_symmetry(video)
        self.check_reconstruction(video)

    def test_signal_rec_diff_odd_dim(self):

        video = self.get_random_video(3, 1, 3)
        self.check_fourier_symmetry(video)
        self.check_reconstruction(video)
        video = self.get_random_video(3, 1, 2)
        self.check_fourier_symmetry(video)
        self.check_reconstruction(video)

    def test_signal_rec_same_even_dim(self):
        video = self.get_random_video(2, 2, 2)
        self.check_fourier_symmetry(video)
        self.check_reconstruction(video)
        video = self.get_random_video(4, 4, 2)
        self.check_fourier_symmetry(video)
        self.check_reconstruction(video)

    def test_signal_rec_mixed_dim(self):
        video = self.get_random_video(3, 4, 2)
        self.check_fourier_symmetry(video)
        self.check_reconstruction(video)
        video = self.get_random_video(4, 5, 3)
        self.check_fourier_symmetry(video)
        self.check_reconstruction(video)

    def test_signal_rec_sym_samples(self):
        np.random.seed(0)
        video = self.get_random_video(4, 4, 4)
        self.check_fourier_symmetry(video)
        self.check_reconstruction(video, 0.4)

    def get_random_video(self, VID_WIDTH, VID_HEIGHT, num_images):
        TD = np.random.random(size=(VID_HEIGHT, VID_WIDTH, num_images))
        opt = {"time_domain_samples": TD}
        video = MultiDimPeriodicSignal(opt)
        return video

    def check_fourier_symmetry(self, video):
        f_s = video.freq_domain_samples
        # print("FS",f_s)

        for i in np.arange(f_s.shape[0]):
            for j in range(f_s.shape[1]):
                for k in range(f_s.shape[2]):
                    if i + j + k > 0:
                        assert f_s[i, j, k] == f_s[-i, -j, -k].conj()

    def check_reconstruction(self, video, offset=0.1):

        TEM_locations = []
        hor_separation, ver_separation = 1, 1
        hor_loc_range = np.arange(offset, video.periods[1], hor_separation)
        ver_loc_range = np.arange(offset, video.periods[0], ver_separation)
        for h in hor_loc_range:
            for v in ver_loc_range:
                TEM_locations.append([v, h])

        num_spikes = 8

        signals = SignalCollection.periodicBandlimitedSignals(period=video.periods[-1])
        deltas = []

        for TEM_l in TEM_locations:
            signal_l = video.get_time_signal(TEM_l)
            signals.add(signal_l)
            deltas.append(
                signal_l.get_precise_integral(0, video.periods[-1]) / (2 * num_spikes)
            )

        kappa, b = 1, 0
        tem_mult = TEMParams(kappa, deltas, b, np.eye(len(TEM_locations)))
        end_time = video.periods[-1]
        spikes = Encoder.ContinuousEncoder(tem_mult).encode(
            signals, end_time, tolerance=1e-14, with_start_time=False
        )
        # spikes = ContinuousEncoder(tem_mult).encode_video(video, TEM_locations,end_time, tol=1e-14, with_start_time = False)

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
        coefficients = np.reshape(coefficients, video.num_components)
        print("ERROR", np.log(np.real(coefficients - video.freq_domain_samples) ** 2))

        print("REC", coefficients)

        print("ORIG", video.freq_domain_samples)
        error = np.linalg.norm(
            (coefficients.flatten() - video.freq_domain_samples.flatten())
        ) / np.linalg.norm((video.fft.flatten()))
        assert error < 1e-4


if __name__ == "__main__":

    TestMultiDimPeriodicSignal().test_signal_sampling_nonint4()
    TestMultiDimPeriodicSignal().test_time_signal_object_creation()
    TestMultiDimPeriodicSignal().test_signal_integration()
    TestMultiDimPeriodicSignal().test_signal_rec_same_even_dim()
