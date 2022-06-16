from src import *


class _MultiDimSignal(object):
    def __init__(self, opt):
        self.numDimensions = opt.numDimensions


class MultiDimPeriodicSignal(_MultiDimSignal):
    """
    Implements a multi dimensional signal which is periodic.
    Main use is to represent video
    """

    # Note that freq_domain_samples are assumed to be FFT samples (so the complex conjugate counterpart is not explicitly included in the variable)
    def __init__(self, opt):
        if "time_domain_samples" in opt:
            self.time_domain_samples = opt["time_domain_samples"]
            self.fft = np.fft.fftn(
                self.time_domain_samples
            )  # , s = [int(s/2)*2+1 for s in self.time_domain_samples.shape])
        elif "freq_domain_samples" in opt:
            self.fft = opt["freq_domain_samples"]
            self.time_domain_samples = np.fft.ifftn(self.fft)
        elif "fs_components" in opt:
            raise NotImplementedError
        else:
            raise NotImplementedError
        self.numDimensions = len(self.time_domain_samples.shape)
        self.periods = self.time_domain_samples.shape
        self.n_t_components = int(self.periods[-1] / 2) + 1
        self.dim_frequencies = []
        for dim in range(self.numDimensions):
            self.dim_frequencies.append(self.__get_dim_frequencies(dim))
        self.num_components = [
            len(self.dim_frequencies[nD]) for nD in range(self.numDimensions)
        ]
        self.freq_domain_samples = self.__adjust_freq_domain_samples(self.fft)
        self.__set_fs_components()

    def __get_dim_frequencies(self, dim):
        dim_frequencies = 1 / self.periods[dim] * np.arange(0, self.periods[dim], 1)
        dim_frequencies = 2 * np.pi * (np.remainder(dim_frequencies + 0.5, 1) - 0.5)
        if self.periods[dim] % 2 == 0:
            dim_frequencies = np.insert(
                dim_frequencies, int(self.periods[dim] / 2), np.pi
            )

        return dim_frequencies

    def __adjust_freq_domain_samples(self, freq_samples):
        adjusted_freq_samples = copy.deepcopy(freq_samples)
        for nD in range(self.numDimensions):
            ind = [slice(0, None)] * self.numDimensions
            ind[nD] = int(self.periods[nD] / 2)

            if freq_samples.shape[nD] % 2 == 0:

                adjusted_freq_samples[tuple(ind)] /= 2
                copied_freq_samples = copy.deepcopy(
                    adjusted_freq_samples[tuple(ind)]
                ).conj()
                copied_freq_samples[1:, :] = copied_freq_samples[-1:0:-1, :]
                copied_freq_samples[:, 1:] = copied_freq_samples[:, -1:0:-1]
                adjusted_freq_samples = np.insert(
                    adjusted_freq_samples,
                    int(self.periods[nD] / 2),
                    copied_freq_samples,
                    axis=nD,
                )

        return adjusted_freq_samples

    def __set_fs_components(self):
        self.fs_components = np.roll(self.freq_domain_samples, self.n_t_components, -1)


    def _get_exponential_nD_factor(self, coordinates, nD):
        component_nD = coordinates[nD] * self.dim_frequencies[nD]
        component_direction_vector = np.ones((self.numDimensions)).astype(int)
        component_direction_vector[nD] = int(self.num_components[nD])
        return np.reshape(component_nD, component_direction_vector)

    def _get_exponentials_vector(self, coordinates):
        exponents = np.zeros_like(self.freq_domain_samples, dtype="float")
        for nD in range(self.numDimensions):
            exponents += self._get_exponential_nD_factor(coordinates, nD)
        return np.exp(1j * exponents)

    def sample(self, coordinates):
        exponentials = self._get_exponentials_vector(coordinates)
        weighted_exponentials = np.multiply(self.freq_domain_samples, exponentials)
        return np.real(1 / (np.product(self.periods)) * np.sum(weighted_exponentials))


