import numpy as np
from scipy.special import sici


def Si(t, Omega):
    return sici(Omega * t)[0] / np.pi


def Sii(t, Omega):
    return (np.cos(Omega * t) + Omega * t * sici(Omega * t)[0]) / (Omega * np.pi)


def Di(t, period, n_components):
    component_range = np.arange(1, n_components)
    integral = 1 / period * t
    for n, component in enumerate(component_range):
        integral += (
            2
            * period
            / (2 * np.pi * component)
            * np.sin(2 * np.pi / period * component * t)
        )
    return integral


def Dii(t, period, n_components):
    component_range = np.arange(1, n_components)
    integral = 1 / (2 * period) * t ** 2
    for n, component in enumerate(component_range):
        integral -= (
            2
            * (period / (2 * np.pi * component)) ** 2
            * np.cos(2 * np.pi / period * component * t)
        )
    return integral


def sinc(t, Omega):
    return np.sinc(Omega / np.pi * t) * Omega / np.pi


def exp_int(exponent, t_start, t_end, tolerance=1e-18):
    assert len(t_start) == len(
        t_end
    ), "You should have as many end times as start times for the integrals of the exponentials"
    # exponent = np.atleast_2d(exponent).T
    integrals = np.zeros((len(exponent), len(t_start)), dtype=np.complex_)
    t_start = np.atleast_2d(t_start)
    t_end = np.atleast_2d(t_end)
    for n, exp_n in enumerate(exponent):
        if np.abs(exp_n) > tolerance:
            integrals[n, :] = (np.exp(exp_n * t_end) - np.exp(exp_n * t_start)) / exp_n
        else:
            integrals[n, :] = t_end - t_start
    return integrals
