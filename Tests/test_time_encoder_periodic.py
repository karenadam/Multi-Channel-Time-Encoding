import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/..")

from src import *


class TestTimeEncoderPeriodicWithStructure:
    def test_single_signal_single_channel_bandlimited_period_encoding(self):
        kappa = 1
        delta = 1
        b = 1

        period = 3
        signal = src.signals.periodicBandlimitedSignal(period, 2, [1, 2])
        signals = src.signals.periodicBandlimitedSignals(period)
        signals.add(signal)
        delta_t = 1e-4
        time = np.arange(0, 3, delta_t)
        y = signal.sample(time)

        b = np.max(np.abs(y)) + 1

        tem_params = TEMParams(kappa, delta, b, mixing_matrix=[[1]])
        spikes_single = encoder.DiscreteEncoder(tem_params).encode(
            signal, signal_end_time=3, delta_t=delta_t
        )

        spikes_single_precise = encoder.ContinuousEncoder(tem_params).encode(
            signals, time[-1]
        )

        spikes_single_precise = encoder.ContinuousEncoder(tem_params).encode_with_disc_hot_start(
            signals, time[-1]
        )

        print(spikes_single[0])
        print(spikes_single_precise[0])
        assert np.allclose(spikes_single[0], spikes_single_precise[0], atol = 1e-3)

    def test_ss_sc_q_generation(self):
        kappa = 1
        delta = 1
        b = 1
        period = 3
        n_components = 2

        spikes = SpikeTimes(n_channels=1)
        spikes.add(0, [0.5, 2, 2.5, 3])

        tem_params = TEMParams(kappa, delta, b, mixing_matrix=[[1]])
        q = decoder.SSignalMChannelDecoder(
            tem_params, periodic=True, period=period, n_components=n_components
        ).get_measurement_vector(spikes)
        G = decoder.SSignalMChannelDecoder(
            tem_params, periodic=True, period=period, n_components=n_components
        ).get_measurement_operator(
            spikes,
        )
        target_q = [0.5, 1.5, 1.5]
        assert np.allclose(q, target_q)

    def test_ss_sc_bl_decoding(self):
        kappa = 1
        delta = 0.5
        b = 2
        period = 3
        n_components = 3
        delta_t = 1e-4
        time = np.arange(0, 3, delta_t)

        signal = src.signals.periodicBandlimitedSignal(period, n_components, [1, 2, -1])
        total_integral = signal.get_precise_integral(0, 3)
        signals = src.signals.periodicBandlimitedSignals(
            period, coefficient_values=[]
        )
        signals.add(signal)
        tem_params = TEMParams(
            kappa, delta, b, mixing_matrix=[[1]], integrator_init=[delta]
        )
        spikes_single = encoder.ContinuousEncoder(tem_params).encode(signals, period)
        y = signal.sample(time)

        spikes = SpikeTimes(n_channels=1)

        rec_single = decoder.SSignalMChannelDecoder(
            tem_params, periodic=True, period=period, n_components=n_components
        ).decode(spikes_single, time)
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)

        assert (
            np.mean(((np.real(rec_single) - np.real(y)) ** 2)[start_index:end_index])
            / np.mean(np.real(y) ** 2)
            < 1e-3
        )

    def test_ss_2c_bl_decoding(self):
        kappa = 1
        delta = 1
        b = 2
        period = 3
        n_components = 3
        delta_t = 1e-4
        time = np.arange(0, 3, delta_t)

        signal = src.signals.periodicBandlimitedSignal(period, n_components, [1, 2, -1])
        total_integral = signal.get_precise_integral(0, 3)
        signals = src.signals.periodicBandlimitedSignals(
            period, coefficient_values=[]
        )
        signals.add(signal)
        tem_params = TEMParams(
            kappa, delta, b, mixing_matrix=[[1], [1]], integrator_init=[-delta, 0]
        )
        spikes_single = encoder.ContinuousEncoder(tem_params).encode(signals, period)
        y = signal.sample(time)

        spikes = SpikeTimes(n_channels=1)

        rec_single = decoder.SSignalMChannelDecoder(
            tem_params, periodic=True, period=period, n_components=n_components
        ).decode(
            spikes_single,
            time,
        )
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)

        assert (
            np.mean(((np.real(rec_single) - np.real(y)) ** 2)[start_index:end_index])
            / np.mean(np.real(y) ** 2)
            < 1e-3
        )

    def test_2s_3c_bl_decoding(self):
        kappa = 1
        delta = 1
        b = 2
        period = 3
        n_components = 3
        delta_t = 1e-4
        time = np.arange(0, 3, delta_t)

        signal = src.signals.periodicBandlimitedSignal(period, n_components, [1, 2, -1])
        signal2 = src.signals.periodicBandlimitedSignal(period, n_components, [-1, 0, 3])
        signals = src.signals.periodicBandlimitedSignals(
            period, coefficient_values=[]
        )
        signals.add(signal)
        signals.add(signal2)
        tem_params = TEMParams(
            kappa, delta, b, mixing_matrix=[[1, -2], [1, 0.5], [0.1, 0.7]]
        )
        spikes = encoder.ContinuousEncoder(tem_params).encode(signals, period)
        y = signal.sample(time)

        rec = decoder.MSignalMChannelDecoder(
            tem_params, periodic=True, period=period, n_components=n_components
        ).decode(spikes, time)
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)

        assert (
            np.mean(((np.real(rec[0, :]) - np.real(y)) ** 2)[start_index:end_index])
            / np.mean(np.real(y) ** 2)
            < 1e-3
        )


class TestTimeEncoderBandlimitedPeriodicWithStructure:
    def test_multi_signal_multi_channel_bandlimited_period_one_shot_reconstruction_using_old_rec_alg(
        self,
    ):
        kappa = [0.25, 0.5, 0.5]
        delta = [1, 1, 1]
        b = 1

        period = 3
        n_components = 3
        np.random.seed(77)
        signal = src.signals.periodicBandlimitedSignal(period, n_components, [1, 3, -1])
        signal2 = src.signals.periodicBandlimitedSignal(period, n_components, [2, 1, 1])
        signal3 = src.signals.periodicBandlimitedSignal(period, n_components, [-1, 2, 0])
        signals = src.signals.periodicBandlimitedSignals(
            period, coefficient_values=[]
        )
        signals.add(signal)
        signals.add(signal2)
        signals.add(signal3)
        delta_t = 1e-4
        time = np.arange(0, 3, delta_t)
        # y = signal.sample(time)
        mixing_matrix = np.array([[1, 2, 0], [0, 1, 1], [0, 0, 1]])
        mixed = signals.get_mixed_signals(mixing_matrix)
        y = signals[0].sample(time)

        b = np.max(np.abs(y)) + 1

        tem_params = TEMParams(kappa, delta, b, mixing_matrix)

        spikes_single_precise = encoder.ContinuousEncoder(tem_params).encode(
            signals, time[-1]
        )
        # spikes_single_precise.print()

        spikes_0 = spikes_single_precise.get_spikes_of(0)
        sinc_locs = (spikes_0[:-1] + spikes_0[1:]) / 2

        rec_single = decoder.MSignalMChannelDecoder(
            tem_params, periodic=False, sinc_locs=sinc_locs, Omega=2 * np.pi / period
        ).decode(
            spikes_single_precise,
            time,
        )

        # rec_single = tem_single.decode_periodic(spikes_single_precise, time, period, n_components, delta_t)
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)

        assert (
            np.mean(((np.real(rec_single[0]) - np.real(y)) ** 2)[start_index:end_index])
            / np.mean(np.real(y) ** 2)
            < 1e-3
        )

    def test_1Dspace_BL_signal_multi_channel_bandlimited_period_one_shot_reconstruction_using_old_rec_alg(
        self,
    ):
        kappa = [0.25, 0.5, 0.5, 1]
        delta = [1, 1, 1, 1]
        b = 1

        period = 3
        n_components = 3
        np.random.seed(77)
        signal = src.signals.periodicBandlimitedSignal(period, n_components, [1, 3, -1])
        signal2 = src.signals.periodicBandlimitedSignal(period, n_components, [2, 1, 1])
        signal3 = src.signals.periodicBandlimitedSignal(period, n_components, [-1, 2, 0])
        signals = src.signals.periodicBandlimitedSignals(
            period, coefficient_values=[]
        )
        signals.add(signal)
        signals.add(signal2)
        signals.add(signal3)
        delta_t = 1e-4
        time = np.arange(0, 3, delta_t)
        # y = signal.sample(time)
        spatial_period = 6
        spatial_n_components = signals.n_signals
        spatial_frequencies = np.atleast_2d(
            2 * np.pi / spatial_period * np.arange(0, spatial_n_components, 1)
        )
        spatial_sample_locations = np.atleast_2d([1, 2, 3, 5]).T
        mixing_matrix = np.real(
            np.exp(-1j * spatial_frequencies * spatial_sample_locations)
        )
        mixed = signals.get_mixed_signals(mixing_matrix)
        y = signals[0].sample(time)
        print(y)

        b = np.max(np.abs(y)) + 1

        tem_params = TEMParams(kappa, delta, b, mixing_matrix)

        spikes_single_precise = encoder.ContinuousEncoder(tem_params).encode(
            signals, time[-1]
        )
        # spikes_single_precise.print()

        spikes_0 = spikes_single_precise.get_spikes_of(0)
        sinc_locs = (spikes_0[:-1] + spikes_0[1:]) / 2

        rec_single = decoder.MSignalMChannelDecoder(
            tem_params, periodic=False, sinc_locs=sinc_locs, Omega=2 * np.pi / period
        ).decode(
            spikes_single_precise,
            time,
        )

        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)

        assert (
            np.mean(((np.real(rec_single[0]) - np.real(y)) ** 2)[start_index:end_index])
            / np.mean(np.real(y) ** 2)
            < 1e-2
        )

        # continuous_image_time = np.arange(0,1.5*period, 1e-2 )
        # continuous_image_sample_locs = np.atleast_2d(np.arange(0,2*spatial_period,0.1)).T
        # continuous_image_mixing_matrix = np.real(np.exp(-1j*spatial_frequencies*continuous_image_sample_locs))
        # continuous_image = signals.get_mixed_signals(continuous_image_mixing_matrix).sample(continuous_image_time)
        # print(continuous_image.shape)

        # plt.figure()
        # sns.heatmap(continuous_image[:,0:1].T)#, xticklabels = continuous_image_sample_locs, yticklabels = time)
        # # plt.ylabel('Space')
        # plt.xlabel('Space')
        # plt.savefig('cont_1D_im.png')
        # assert False

        # b = np.max(np.abs(y)) + 1

    # def test_single_signal_multi_channel_bandlimited_period_one_shot_reconstruction(self):
    #     assert False

    # def test_multi_signal_multi_channel_bandlimited_period_one_shot_reconstruction(self):
    #     assert False

    # def test_1D_scene_bandlimited_period_one_shot_reconstruction(self):
    #     assert False

    # def test_2D_scene_bandlimited_period_one_shot_reconstruction(self):
    #     assert False


if __name__ == "__main__":

    TestTimeEncoderPeriodicPeriodicWithStructure().test_single_signal_single_channel_bandlimited_period_encoding()
