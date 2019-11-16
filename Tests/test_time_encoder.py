import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/../Source")
from Time_Encoder import timeEncoder
from Signal import bandlimitedSignal


class TestTimeEncoderSingleSignalSingleChannel:
    def test_time_encoder_creator(self):
        TEM = timeEncoder(kappa=1, delta=1, b=1, mixing_matrix=[[1]])
        assert TEM.kappa == [1], "kappa was falsely assigned"
        assert TEM.delta == [1], "delta was falsely assigned"
        assert TEM.b == [1], "b was falsely assigned"

        TEM2 = timeEncoder(kappa=[1], delta=1, b=1, mixing_matrix=[[1],[1]])
        assert len(TEM2.kappa) == 2

    def test_can_reconstruct_standard_encoding(self):
        kappa = 1
        delta = 1

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 15, delta_t)
        np.random.seed(10)
        original = bandlimitedSignal(t, delta_t, omega)
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1

        tem_single = timeEncoder(kappa, delta, b, mixing_matrix=[[1]])
        spikes_single = tem_single.encode(y, delta_t)
        rec_single = tem_single.decode(spikes_single, t, omega, delta_t)
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
        original = bandlimitedSignal(t, delta_t, omega)
        x_param.append(original)
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1

        tem_single = timeEncoder(kappa, delta, b, mixing_matrix = [[1]])
        spikes_single = tem_single.encode_precise(
            original, omega, t[-1]
        )
        rec_single = tem_single.decode(spikes_single, t, omega, delta_t)
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)
        assert (
            np.mean(((rec_single - y) ** 2)[start_index:end_index]) / np.mean(y ** 2)
            < 1e-3
        )


class TestTimeEncoderSingleSignalMultiChannel:
    def test_can_reconstruct_standard_encoding_ex1(self):
        kappa = [4, 3.2, 2.8, 3]
        delta = [0.8, 1.2, 0.6, 1]
        int_shift = max(delta) / 2

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 20, delta_t)
        np.random.seed(10)
        original = bandlimitedSignal(t, delta_t, omega)
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1

        tem_mult = timeEncoder(
            kappa, delta, b, mixing_matrix=[[1]]*4, integrator_init=[int_shift]
        )
        spikes_mult = tem_mult.encode(y, delta_t)
        rec_mult = tem_mult.decode(spikes_mult, t, omega, delta_t)
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)
        assert (
            np.mean(((rec_mult - y) ** 2)[start_index:end_index]) / np.mean(y ** 2)
            < 1e-3
        )

    def test_TEM_can_reconstruct_standard_encoding_ex2(self):
        kappa = [1, 1, 1, 1]
        delta = [1, 1, 1, 1]
        int_shift = [-1, -0.5, 0, 0.5]

        omega = 2*np.pi
        delta_t = 1e-4
        t = np.arange(0, 20, delta_t)
        np.random.seed(10)
        original = bandlimitedSignal(t, delta_t, omega)
        y = original.sample(t)
        b = np.max(np.abs(y)) + 1

        tem_mult = timeEncoder(kappa, delta, b, mixing_matrix=[[1]]*4, integrator_init=int_shift)
        spikes_mult = tem_mult.encode(y, delta_t)
        rec_mult = tem_mult.decode(spikes_mult, t, omega, delta_t)
        start_index = int(len(y) / 10)
        end_index = int(len(y) * 9 / 10)
        assert (
            np.mean(((rec_mult - y) ** 2)[start_index:end_index]) / np.mean(y ** 2)
            < 1e-3
        )

    def test_can_reconstruct_standard_encoding_with_recursive_mixing_alg(self):
        kappa = [1, 1]
        delta = [2, 1]
        b = 1.5
        int_shift = [-1, -0.1]

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 25, delta_t)
        np.random.seed(10)
        original = bandlimitedSignal(t, delta_t, omega)
        y = original.sample(t)
        y = np.atleast_2d(y)
        A = [[1], [2]]

        tem_mult = timeEncoder(kappa, delta, b, A, integrator_init=int_shift)
        spikes_mult = tem_mult.encode(y, delta_t)
        rec_mult = tem_mult.decode_recursive(
            spikes_mult, t, omega, delta_t, num_iterations=20
        )

        start_index = int(y.shape[1] / 10)
        end_index = int(y.shape[1] * 9 / 10)

        assert (
            np.mean(((rec_mult[0, :] - y[0, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )


class TestTimeEncoderMultiSignalMultiChannel:
    def test_can_compute_q_matrix(self):
        kappa = [1, 1]
        delta = [2, 1]
        b = 2
        int_shift = [-1, -0.1]

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 25, delta_t)
        np.random.seed(10)
        original = bandlimitedSignal(t, delta_t, omega)
        y = original.sample(t)
        y = np.atleast_2d(y)
        A = [[1], [2]]

        # Compute q matrix in 2 ways and make sure they match
        tem_mult = timeEncoder(kappa, delta, b, A, integrator_init=int_shift)
        spikes_mult = tem_mult.encode(y, delta_t)
        q1, G = tem_mult.get_closed_form_matrices(
            spikes_mult, omega
        )
        q2 = tem_mult.get_integrals(np.array(A).dot(y), spikes_mult, delta_t, q1.shape)
        assert np.mean((q1 - q2) ** 2) < (delta_t) ** 2

    def test_can_reconstruct_standard_encoding_with_2_by_2_mixing_recursive(self):
        kappa = [1, 1]
        delta = [2, 1]
        b = [5, 3]
        int_shift = [-1, -0.1]

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 25, delta_t)
        np.random.seed(10)
        original1 = bandlimitedSignal(t, delta_t, omega)
        np.random.seed(11)
        original2 = bandlimitedSignal(t, delta_t, omega)
        y = np.zeros((2, len(t)))
        y[0, :] = original1.sample(t)
        y[1, :] = original2.sample(t)
        y = np.atleast_2d(y)
        A = [[0.9, 0.1], [0.2, 0.8]]

        tem_mult = timeEncoder(kappa, delta, b, A, integrator_init=int_shift)
        spikes_mult = tem_mult.encode(y, delta_t)
        rec_mult = tem_mult.decode_recursive(
            spikes_mult, t, omega, delta_t, num_iterations=20
        )

        start_index = int(y.shape[1] / 10)
        end_index = int(y.shape[1] * 9 / 10)

        assert (
            np.mean(((rec_mult[0, :] - y[0, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )
        assert (
            np.mean(((rec_mult[1, :] - y[1, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )

    def test_can_reconstruct_standard_encoding_with_3_by_2_mixing_recursive(self):
        kappa = [1, 1, 1]
        delta = [2, 1, 1]
        b = [2, 2, 2]
        int_shift = [-1, -0.1, 0.2]

        omega = np.pi
        delta_t = 1e-4
        t = np.arange(0, 25, delta_t)
        np.random.seed(10)
        original1 = bandlimitedSignal(t, delta_t, omega)
        np.random.seed(11)
        original2 = bandlimitedSignal(t, delta_t, omega)
        y = np.zeros((2, len(t)))
        y[0, :] = original1.sample(t)
        y[1, :] = original2.sample(t)
        y = np.atleast_2d(y)
        A = [[0.9, 0.1], [0.2, 0.8], [1, 1]]

        tem_mult = timeEncoder(kappa, delta, b, A, integrator_init=int_shift)
        spikes_mult = tem_mult.encode(y, delta_t)
        rec_mult = tem_mult.decode_recursive(
            spikes_mult, t, omega, delta_t, num_iterations=20
        )

        start_index = int(y.shape[1] / 10)
        end_index = int(y.shape[1] * 9 / 10)

        assert (
            np.mean(((rec_mult[0, :] - y[0, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )
        assert (
            np.mean(((rec_mult[1, :] - y[1, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )


    def test_can_reconstruct_precise_encoding_with_3_by_2_mixing_recursive(self):
        kappa = [1, 1, 1]
        delta = [2, 1, 1]
        b = [2, 2, 2]
        int_shift = [-1, -0.1, 0.2]

        omega = np.pi
        delta_t = 1e-4
        end_time = 25
        t = np.arange(0, 25, delta_t)
        y_param = []
        np.random.seed(10)
        original1 = bandlimitedSignal(t, delta_t, omega)
        np.random.seed(11)
        original2 = bandlimitedSignal(t, delta_t, omega)
        y = np.zeros((2, len(t)))
        y[0, :] = original1.sample(t)
        y[1, :] = original2.sample(t)
        y = np.atleast_2d(y)
        A = [[0.9, 0.1], [0.2, 0.8], [1, 1]]
        y_param.append(original1)
        y_param.append(original2)

        tem_mult = timeEncoder(kappa, delta, b, A, integrator_init=int_shift)
        spikes_mult = tem_mult.encode_precise(y_param, omega, end_time,)
        rec_mult = tem_mult.decode_recursive(
            spikes_mult, t, omega, delta_t, num_iterations=20
        )

        start_index = int(y.shape[1] / 10)
        end_index = int(y.shape[1] * 9 / 10)

        assert (
            np.mean(((rec_mult[0, :] - y[0, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )
        assert (
            np.mean(((rec_mult[1, :] - y[1, :]) ** 2)[start_index:end_index])
            / np.mean(y[0, :] ** 2)
            < 1e-3
        )
    def test_params_TEM_correct(self):
        TEM = timeEncoder(kappa=1, delta=1, b=1, mixing_matrix=[[1]]*2)
        assert TEM.kappa == [1, 1], "kappa was falsely assigned"
        assert TEM.delta == [1, 1], "delta was falsely assigned"
        assert TEM.b == [1, 1], "b was falsely assigned"
