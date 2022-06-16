from src import *

# import Signals
from typing import Union


class FRISignal(object):
    """
    periodic finite rate of innovation signals made of a sum of diracs

    ATTRIBUTES
    ----------
    _num_diracs: int
        number of diracs in signal
    _dirac_locations: list
        list of floats representing locations of diracs
    _dirac_weights: list
        list of floats representing weights of diracs
    _period: float
        period of signal
    """

    def __init__(
        self, dirac_locations: np.array, dirac_weights: np.array, period: float
    ):
        """
        PARAMETERS
        ----------
        dirac_locations: list or np.ndarray
            list of floats representing locations of diracs
        dirac_weights: list
            list of floats representing weights of diracs
        period: float
            period of signal
        """

        self._num_diracs = len(dirac_locations)
        self._dirac_locations = np.atleast_2d(copy.deepcopy(dirac_locations))
        self._dirac_weights = np.atleast_2d(copy.deepcopy(dirac_weights))
        self._period = period
        return

    def get_fourier_series(self, m: np.array):
        """
        PARAMETERS
        ----------
        m: np.ndarray
            indices of desired fourier series coefficients

        RETURNS
        -------
        np.ndarray
            fourier series coefficients of indices m
        """
        m = np.reshape(m, (-1, 1))
        c_k_m = self._dirac_weights * np.exp(
            -1j * 2 * np.pi * m * self._dirac_locations / self._period
        )
        f_s_m = 1 / self._period * np.sum(c_k_m, 1)
        return f_s_m


class AnnihilatingFilter(object):
    def check_coefficient_shape_and_structure(self, f_s_coefficients: np.array):
        if len(f_s_coefficients.shape) == 1:
            f_s_coefficients = np.reshape(
                f_s_coefficients, (1, f_s_coefficients.shape[0])
            )
        if not np.allclose(f_s_coefficients, f_s_coefficients[:, ::-1].conj()):
            raise ValueError("Coefficients provided are not conjugate symmetric")
        return f_s_coefficients

    def __init__(self, f_s_coefficients: np.array, filter_length: int = 1):
        f_s_coefficients = self.check_coefficient_shape_and_structure(f_s_coefficients)

        num_available_fs_coeffs = int((f_s_coefficients.shape[1] + 1) / 2)
        if filter_length == 1:
            self._num_taps = num_available_fs_coeffs
        else:
            self._num_taps = filter_length

        self._num_annihilated_signals = f_s_coefficients.shape[0]

        if num_available_fs_coeffs >= self._num_taps:
            extended_f_s_coefficients = f_s_coefficients
        else:
            # TODO this would be the case of joint annihilation.. something is wrong
            extended_f_s_coefficients = np.zeros(
                (self._num_annihilated_signals, 2 * self._num_taps - 1), dtype="complex"
            )
            extended_f_s_coefficients[
                :,
                self._num_taps
                - num_available_fs_coeffs : self._num_taps
                + num_available_fs_coeffs
                - 1,
            ] = f_s_coefficients[:, :]

        num_rotations = self._num_taps - 2

        operator = np.zeros(
            ((num_rotations) * self._num_annihilated_signals + 1, self._num_taps - 1),
            dtype="complex",
        )
        measurements = np.zeros((operator.shape[0], 1), dtype="complex")

        for n_s in range(self._num_annihilated_signals):
            for n_t in range(num_rotations):
                operator[n_s * num_rotations + n_t, :] = extended_f_s_coefficients[
                    n_s, self._num_taps + n_t - 1 : n_t : -1
                ]
                measurements[n_s * num_rotations + n_t] = -extended_f_s_coefficients[
                    n_s, self._num_taps + n_t
                ]

        self._filter_coefficients = np.ones((self._num_taps, 1), dtype="complex")
        self._filter_coefficients[1:] = np.linalg.lstsq(
            operator, measurements, rcond=None
        )[0]
        self._annihilation_operator = self.get_annihilation_operator()

    def get_filter_coefficients(self):
        return self._filter_coefficients.flatten()

    def get_annihilation_operator(self):
        num_annihilation_constraints = self._num_taps - 1
        num_real_signal_constraints = self._num_taps - 1
        operator = np.zeros(
            (
                num_annihilation_constraints + num_real_signal_constraints,
                self._num_taps * 2 - 1,
            ),
            dtype="complex",
        )
        for n_a_c in range(num_annihilation_constraints):
            operator[
                n_a_c, n_a_c + 1 : self._num_taps + 1 + n_a_c
            ] = self._filter_coefficients[::-1].flatten()
        for n_r_c in range(num_real_signal_constraints):
            operator[num_annihilation_constraints + n_r_c, :] = operator[
                n_r_c, ::-1
            ].conj()
        return operator

    def make_annihilatable(self, fri_signal: Union[FRISignal, np.array]):
        if isinstance(fri_signal, FRISignal):
            f_s = fri_signal.get_fourier_series(
                np.arange(-self._num_taps + 1, self._num_taps, 1).T
            )
        else:
            f_s = fri_signal
        annihilatable_f_s = f_s - np.linalg.pinv(self._annihilation_operator).dot(
            self._annihilation_operator.dot(f_s)
        )
        return annihilatable_f_s
