from src import *


class TEMParams(object):
    """
    A class used to represent parameters of a, or multiple, integrate-
    and-fire time encoding machine(s)

    Attributes
    ----------
    n_signals: int
        number of signals the time encoding machines encode
    n_channels: int
        number of channels used for the encoding. this can be different
        from the number of signals, in which case there is some mixing
        of the signals before they are fed into the channels
    kappa: list
        list of floats that represent integrator constants of each TEM
    delta: list
        list of floats that represent firing threshold of each TEM
    b: list of floats
        list of floats that represent bias added to input signal for each TEM
    mixing_matrix: np.ndarray
        matrix A where y = Ax where x is the input to the system and x
        is the vector of inputs to the time encoding machines
    integrator_init: list
        list of float values that specify the value of the integrator
        at start of the integration
    """

    def __init__(
        self,
        kappa,
        delta,
        b,
        mixing_matrix,
        integrator_init=[],
    ):
        """
        Makes sure all parameters have the right shapes before initializing

        Parameters
        ----------
        kappa: float
            integrator constant
        delta: float
            firing threshold
        b: float
            bias added to input signal
        mixing_matrix: np.ndarray
            matrix A where y = Ax where x is the input to the system and x
            is the vector of inputs to the time encoding machines
        integrator_init: list
            list of float values that specify the value of the integrator
            at start of the integration
        """

        self.mixing_matrix = np.atleast_2d(np.array(mixing_matrix))
        self.n_signals = self.mixing_matrix.shape[1]
        self.n_channels = self.mixing_matrix.shape[0]
        self._kappa = self._check_dimensions(kappa)
        self._delta = self._check_dimensions(delta)
        if len(integrator_init) > 0:
            self._integrator_init = self._check_dimensions(integrator_init)
        else:
            self._integrator_init = [-self.delta[l] for l in range(self.n_channels)]
        self._b = self._check_dimensions(b)

    def getKappa(self):
        return self._kappa

    def setKappa(self, kappa):
        self._kappa = self._check_dimensions(kappa)

    def getDelta(self):
        return self._delta

    def setDelta(self, delta):
        self._delta = self._check_dimensions(delta)

    def getB(self):
        return self._b

    def setB(self, b):
        self._b = self._check_dimensions(b)

    def getIntegratorInit(self):
        return self._integrator_init

    def setIntegratorInit(self, integrator_init):
        self._integrator_init = self._check_dimensions(integrator_init)

    kappa = property(getKappa, setKappa)
    delta = property(getDelta, setDelta)
    integrator_init = property(getIntegratorInit, setIntegratorInit)
    b = property(getB, setB)

    def __repr__(self):
        return (
            "TEMParams with "
            + str(self.n_signals)
            + " signals, "
            + str(self.n_channels)
            + " channels , kappa = "
            + str(self.kappa)
            + ", delta = "
            + str(self.delta)
            + " b = "
            + str(self.b)
            + ", mixing_matrix = "
            + str(self.mixing_matrix)
            + ", integrator_init = "
            + str(self.integrator_init)
        )

    def _check_dimensions(self, parameter):
        """
        Verifies that parameter has as many entries at the number
        of channels

        Parameters
        ----------
        parameter: list or float
            either a float or a list of floats (for a given parameter
            type kappa, delta, b or integrator_init), each of which corresponds
            to one of the TEM channels

        Raises
        ------
        ValueError
            If the parameter given is a list with a length that is inconsistent
            with the number of channels
        """

        if not isinstance(parameter, list):
            parameter = [parameter] * self.n_channels
        elif len(parameter) == 1:
            parameter = parameter * self.n_channels
        elif len(parameter) != self.n_channels:
            raise ValueError(
                "There should be as many values set for the TEM parameters as there are channels"
            )
        return [float(p) for p in parameter]
