from src import *


class SignalCollection(object):
    """
    Collection of multiple signals of the same type

    ATTRIBUTES
    ----------
    _n_signals: int
        number of signals represented
    _signals: list
        list of Signal objects
    """

    def __init__(self):
        self._n_signals = 0
        self._signals = []

    def add(self, signal):
        """
        PARAMETERS
        ----------
        signal: Signal
            signal to be added to the collection
        """
        self._signals.append(signal)
        self._n_signals += 1

    def sample(self, t):
        """
        samples the different signals in the collection at time(s) t

        PARAMETERS
        ----------
        t: list or np.ndarray
            time(s) at which the signals should be sampled

        Returns
        -------
        np.ndarray
            array where each row represents the samples of one of the signals
            of the collection at time(s) t
        """
        out = np.zeros((self._n_signals, len(t)))
        for n in range(self._n_signals):
            out[n, :] = self._signals[n].sample(t)
        return out

    def __getitem__(self, signal_index):
        return self._signals[signal_index]

    def get_n_signals(self):
        return self._n_signals

    n_signals = property(get_n_signals)


class periodicBandlimitedSignals(SignalCollection):
    """
    Collection of periodic bandlimited signals formed of sums of complex
    exponentials

    ATTRIBUTES
    ----------
    _n_signals: int
        number of signals represented
    _signals: list
        list of periodicBandlimitedSignal objects
    period: float
        period of the signals
    _n_components: int
        number of components in the periodic signals, where the (complex conjugate)
        coefficient values are stored for components -n_components+1 to n_components-1
    _coefficient_values: np.ndarray
        array where each row represents the coefficients of one of the signals
        in the collection
    """

    def __init__(self, period, n_components=0, coefficient_values=None):
        """
        PARAMETERS
        ----------
        period: float
            period of the signals
        n_components: int
            number of components in the periodic signals, where the (complex conjugate)
            coefficient values are stored for components -n_components+1 to n_components-1
        coefficient_values: np.ndarray
            array where each row represents the coefficients of one of the signals
            in the collection
        """

        self.period = period
        self._n_components = n_components
        self._n_signals = (
            len(coefficient_values) if coefficient_values is not None else 0
        )
        self._signals = [
            Signal.periodicBandlimitedSignal(
                period, n_components, coefficient_values[n]
            )
            for n in range(self._n_signals)
        ]
        self._coefficient_values = (
            coefficient_values if coefficient_values is not None else []
        )

    def add(self, signal):
        """
        PARAMETERS
        ----------
        signal: periodicBandlimitedSignal
            signal to be added to collection
        """
        if signal.period != self.period:
            raise ValueError(
                "The period of the signal you are adding does not match the period of the collection"
            )
        if len(self._signals) == 0:
            self._n_components = signal._n_components
        elif self._n_components != signal._n_components:
            raise ValueError(
                "The number of components of the signal you are adding does not match that of the collection"
            )
        self._coefficient_values.append(signal.coefficients)
        super().add(signal)

    def get_mixed_signals(self, mixing_matrix):
        """
        PARAMETERS
        ----------
        mixing_matrix: np.ndarray
            matrix used to mix the coefficients of the collection of the signals

        Returns
        -------
        periodicBandlimitedSignals
            collection of signals with the mixed version of the coefficients of this
            collection (mixed using the provided mixing matrix)
        """
        new_coefficients = mixing_matrix.dot(self._coefficient_values)
        new_signals = periodicBandlimitedSignals(
            self.period, self._n_components, new_coefficients
        )
        return new_signals

    def get_coefficients(self):
        return self._coefficient_values

    coefficient_values = property(get_coefficients)


class bandlimitedSignals(SignalCollection):
    """
    Collection of bandlimited signals formed of sums of sincs

    ATTRIBUTES
    ----------
    _n_signals: int
        number of signals represented
    _signals: list
        list of bandlimitedSignal objects
    _omega: float
        bandwith of the signals, i.e. frequency used for the sincs
    _sinc_locs: np.ndarray
        locations of the sincs that form the signals
    _sinc_amps: np.ndarray
        array where each row represents the amplitudes of the sincs that form the signal
    """

    def __init__(self, Omega, sinc_locs=None, sinc_amps=None):
        """
        PARAMETERS
        ----------
        Omega: float
            bandwith of the signals, i.e. frequency used for the sincs
        sinc_locs: np.ndarray
            locations of the sincs that form the signals
        sinc_amps: np.ndarray
            array where each row represents the amplitudes of the sincs that form the signal
        """
        self._n_signals = len(sinc_amps) if sinc_amps is not None else 0
        self._signals = [
            Signal.bandlimitedSignal(Omega, sinc_locs, sinc_amps[n])
            for n in range(self._n_signals)
        ]
        self._sinc_locs = np.array(sinc_locs) if sinc_locs is not None else []
        self._sinc_amps = sinc_amps if sinc_amps is not None else []
        self._omega = Omega

    def add(self, signal):
        """
        PARAMETERS
        ----------
        signal: bandlimitedSignal
            signal to be added to collection
        """
        if signal.Omega != self._omega:
            raise ValueError(
                "The bandwidth of the signal you are adding does not match the bandwidth of the collection"
            )
        if len(self._signals) == 0:
            self._sinc_locs = copy.deepcopy(signal.get_sinc_locs())
        elif not np.allclose(signal.get_sinc_locs(), self._sinc_locs):
            raise ValueError(
                "Locations of sincs of added signal do not match that of collection"
            )
        self._sinc_amps.append(signal.get_sinc_amps().tolist())
        super().add(signal)

    def mix_amplitudes(self, mixing_matrix):
        """
        PARAMETERS
        ----------
        mixing_matrix: np.ndarray
            matrix used to mix the coefficients of the collection of the signals

        Returns
        -------
        np.ndarray
            flattened version of mixed sinc amplitudes (mixed using mixing_matrix)
        """
        return np.array(mixing_matrix).dot(np.array(self._sinc_amps)).flatten()

    def get_mixed_signals(self, mixing_matrix):
        """
        PARAMETERS
        ----------
        mixing_matrix: np.ndarray
            matrix used to mix the coefficients of the collection of the signals

        Returns
        -------
        bandlimitedSignals
            collection of signals with the mixed version of the coefficients of this
            collection (mixed using the provided mixing matrix)
        """
        new_amps = mixing_matrix.dot(self._sinc_amps)
        new_signals = bandlimitedSignals(self._omega, self._sinc_locs, new_amps)
        return new_signals

    def get_sinc_locs(self):
        return self._sinc_locs

    def get_sinc_amps(self):
        return self._sinc_amps

    def get_omega(self):
        return self._omega

    sinc_locs = property(get_sinc_locs)
    sinc_amps = property(get_sinc_amps)
    omega = property(get_omega)


class piecewiseConstantSignals(SignalCollection):
    """
    Collection of piecewise constant signals

    ATTRIBUTES
    ----------
    _discontinuities: list
        list of lists containing the discontinuities of the different signals
    _values: list
        list of lists containing the values the different signals take between discontinuities
    _n_signals: int
        number of signals in collection
    _signals: list
        list of piecewiseConstantSignal objects
    """

    def __init__(self, discontinuities=[[]], values=[[]]):
        self._discontinuities = discontinuities
        self._values = values
        self._n_signals = len(discontinuities)
        self._signals = [
            Signal.piecewiseConstantSignal(discontinuities[n], values[n])
            for n in range(self._n_signals)
        ]

    def add(self, signal):
        """
        PARAMETERS
        ----------
        signal: piecewiseConstantSignal
            signal to be added to collection
        """
        self.discontinuities.append(signal.get_discontinuities())
        self.values.append(signal.get_values())
        super().add(signal)

    def sample(self, sample_locs, omega):
        """
        samples the signals in the collection at sample_locs and using a low
        pass filter with bandwidth omega

        PARAMETERS
        ----------
        sample_locs: list or np.ndarray
            location(s) at which samples should be taken
        omega: float
            bandwidth of the low pass filter to be used

        Returns
        -------
        np.ndarray
            array where each row represents the samples of one of the signals
            of the collection at time(s) sample_locs
        """
        values = [item for sublist in self.values for item in sublist]
        samples = self.get_sampler_matrix(sample_locs, omega).dot(values)
        return samples

    def get_sampler_matrix(self, sample_locs, omega):
        """
        gets the sampler matrix that will obtain the samples of the signals
        when applied to the flattened version of the list of lists of values

        PARAMETERS
        ----------
        sample_locs: list or np.ndarray
            location(s) at which samples should be taken
        omega: float
            bandwidth of the low pass filter to be used

        Returns
        -------
        np.ndarray
            array where each row obtains a sample of one of the signals at one of the
            sample locations (when applied to the flattened values)
        """

        def multiplier_vector(sample_loc, signal_index):
            low_limit = np.array(self.discontinuities[signal_index][:-1])
            up_limit = np.array(self.discontinuities[signal_index][1:])
            return np.atleast_2d(
                Helpers.sinc_integral(sample_loc - low_limit, omega)
                - Helpers.sinc_integral(sample_loc - up_limit, omega)
            )

        PCS_sampler_matrix = scipy.linalg.block_diag(
            *[
                np.concatenate(
                    [
                        multiplier_vector(sample_loc, signal_index)
                        for sample_loc in sample_locs
                    ]
                )
                for signal_index in range(self._n_signals)
            ]
        )
        return PCS_sampler_matrix

    def get_discontinuities(self):
        return self._discontinuities

    def get_values(self):
        return self._values

    discontinuities = property(get_discontinuities)
    values = property(get_values)
