from src import *


class TEMParams(object):
    """
    A class used to represent parameters of a, or multiple, integrate-
    and-fire time encoding machine(s)

    Attributes
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
    def __init__(
        self,
        kappa,
        delta,
        b,
        mixing_matrix,
        integrator_init=[],
    ):
        self.mixing_matrix = np.atleast_2d(np.array(mixing_matrix))
        self.n_signals = self.mixing_matrix.shape[1]
        self.n_channels = self.mixing_matrix.shape[0]
        self.kappa = self.check_dimensions(kappa)
        self.delta = self.check_dimensions(delta)
        if len(integrator_init) > 0:
            self.integrator_init = self.check_dimensions(integrator_init)
        else:
            self.integrator_init = [-self.delta[l] for l in range(self.n_channels)]
        self.b = self.check_dimensions(b)

    def check_dimensions(self, parameter):
        if not isinstance(parameter, (list)):
            parameter = [parameter] * self.n_channels
        elif len(parameter) == 1:
            parameter = parameter * self.n_channels
        else:
            assert (
                len(parameter) == self.n_channels
            ), "There should be as many values set for the TEM parameters as there are channels"
        return [float(p) for p in parameter]
