import numpy as np
from scipy.special import sici
import numpy.matlib
import bisect
import copy
from Helpers import Si, Sii, sinc, exp_int, Di, Dii
from Signal import *
from Spike_Times import spikeTimes
import time


class TEMParams(object):
    def __init__(
        self,
        kappa,
        delta,
        b,
        mixing_matrix,
        integrator_init=[],
        tol=1e-12,
        precision=10,
    ):
        self.mixing_matrix = np.atleast_2d(np.array(mixing_matrix))
        self.n_signals = self.mixing_matrix.shape[1]
        self.n_channels = self.mixing_matrix.shape[0]
        if isinstance(delta, (list)):
            self.precision = int(precision + 1 / max(delta))
        else:
            self.precision = int(precision + 1 / (delta))
        self.kappa = self.check_dimensions(kappa)
        self.delta = self.check_dimensions(delta)
        if len(integrator_init) > 0:
            self.integrator_init = self.check_dimensions(integrator_init)
        else:
            self.integrator_init = [-self.delta[l] for l in range(self.n_channels)]
        self.b = self.check_dimensions(b)
        self.tol = tol

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
