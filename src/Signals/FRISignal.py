from src import *
from typing import Union

class FRISignal(Signal.Signal):
    def __init__(self, dirac_locations: np.array, dirac_weights: np.array, period: float):
        assert len(dirac_locations.shape) == 1 and len(dirac_weights.shape) == 1, "the dirac locations and weights should be one dimensional arrays"
        assert len(dirac_locations) == len(dirac_weights), "You have a mismatch between numbers of dirac locations and weights"
        self._num_diracs = len(dirac_locations)
        self._dirac_locations = np.atleast_2d(copy.deepcopy(dirac_locations))
        self._dirac_weights = np.atleast_2d(copy.deepcopy(dirac_weights))
        self._period = period
        return
    def get_fourier_series(self, m: np.array):
        if len(m.shape) == 2:
            assert m.shape[1] == 1 # is column vector
        elif len(m.shape) == 1:
            m = np.atleast_2d(m).T
        else:
            assert False, "m does not have the right dimensions"
        c_k_m = self._dirac_weights*np.exp(-1j*2*np.pi*m*self._dirac_locations/self._period)
        f_s_m = 1/self._period * np.sum(c_k_m,1)
        return f_s_m

class AnnihilatingFilter(object):
    def __init__(self, f_s_coefficients: np.array, filter_length: int = 1):
        assert len(f_s_coefficients.shape)<=2
        if f_s_coefficients.shape ==1:
            f_s_coefficients = np.atleast_2d(f_s_coefficients)
        assert (filter_length % 2 == 1)
        f_s_coefficients = np.atleast_2d(f_s_coefficients)
        assert np.allclose(f_s_coefficients, f_s_coefficients[:,::-1].conj()) # Assumes signal is real and checks that coefficients are conj symmetric
        num_taps_per_signal = int((f_s_coefficients.shape[1]+1)/2)
        if filter_length == 1:
            filter_length = num_taps_per_signal
        self._num_annihilated_signals, self._num_taps = f_s_coefficients.shape[0], filter_length
        if f_s_coefficients.shape[1]>2*self._num_taps-1:
            extended_f_s_coefficients = f_s_coefficients
        else:
            extended_f_s_coefficients = np.zeros((self._num_annihilated_signals, 2*self._num_taps-1), dtype = 'complex')
            extended_f_s_coefficients[:, self._num_taps - num_taps_per_signal: self._num_taps + num_taps_per_signal - 1] = f_s_coefficients[:,:]

        # print(extended_f_s_coefficients)
#         Want to solve a system s.t. Xa=Y where X and Y are known
        operator = np.zeros(((num_taps_per_signal-1)*self._num_annihilated_signals,self._num_taps-1), dtype = 'complex')
        measurements = np.zeros((operator.shape[0],1), dtype = 'complex')

        for n_s in range(self._num_annihilated_signals):
            # for n_t in range(self._num_taps-1):
            for n_t in range(num_taps_per_signal-1):
                # print(n_t)
                operator[n_s*(num_taps_per_signal-1)+n_t,:] = extended_f_s_coefficients[n_s,self._num_taps+n_t-1:n_t:-1]
                next_operator = f_s_coefficients[n_s,max(n_t+num_taps_per_signal-self._num_taps,0):num_taps_per_signal+n_t:1][::-1]
                # print(next_operator)
                # print(extended_f_s_coefficients[n_s,self._num_taps+n_t-1:n_t:-1])
                # operator[n_s*(num_taps_per_signal-1)+n_t,:len(next_operator)] = next_operator

                # measurements[n_s*(num_taps_per_signal-1)+n_t] = -extended_f_s_coefficients[n_s,self._num_taps+n_t]
                measurements[n_s*(num_taps_per_signal-1)+n_t] = -f_s_coefficients[n_s,num_taps_per_signal+n_t]


        # print(extended_f_s_coefficients)
        # print("OP: ", operator)
        # print("MEAS: ", measurements)


        self._filter_coefficients = np.ones((self._num_taps,1), dtype = 'complex')
        self._filter_coefficients[1:] = np.linalg.lstsq(operator, measurements, rcond = None)[0]
        self._annihilation_operator = self.get_annihilation_operator()

    def get_filter_coefficients(self):
        return self._filter_coefficients.flatten()

    def get_annihilation_operator(self):
        # print(self._num_taps)
        num_annihilation_constraints = self._num_taps-1
        num_real_signal_constraints = self._num_taps -1
        operator = np.zeros((num_annihilation_constraints + num_real_signal_constraints, self._num_taps*2-1), dtype = 'complex')
        for n_a_c in range(num_annihilation_constraints):
            operator[n_a_c,n_a_c+1:self._num_taps+1+n_a_c] = self._filter_coefficients[::-1].flatten()
        for n_r_c in range(num_real_signal_constraints):
            operator[num_annihilation_constraints+n_r_c, :] = operator[n_r_c,::-1].conj()
        return operator

    def make_annihilatable(self, fri_signal: Union[FRISignal, np.array]):
        if isinstance(fri_signal, FRISignal):
            f_s = fri_signal.get_fourier_series(np.arange(-self._num_taps+1, self._num_taps, 1).T)
        else:
            f_s = fri_signal
        annihilatable_f_s = f_s - np.linalg.pinv(self._annihilation_operator).dot(self._annihilation_operator.dot(f_s))
        return annihilatable_f_s