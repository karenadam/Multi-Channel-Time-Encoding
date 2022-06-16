from src import *


def sinc(t, Omega):
    """
    sinc_Omega(t) = sin(Omega t)/(pi t)

    Parameters
    ----------
    t: float
        time
    Omega: float
        frequency of sinc

    Returns
    -------
    float
        sinc of frequency Omega between time 0 and time t
    """

    return np.sinc(Omega / np.pi * t) * Omega / np.pi


def sinc_integral(t, Omega):
    """
    integral of sinc_Omega(t) = sin(Omega t)/(pi t)

    Parameters
    ----------
    t: float
        time
    Omega: float
        frequency of sinc

    Returns
    -------
    float
        integral of the sinc of frequency Omega between time 0 and time t
    """

    return sici(Omega * t)[0] / np.pi


def sinc_second_integral(t, Omega):
    """
    second integral of sinc_Omega(t) = sin(Omega t)/(pi t)

    Parameters
    ----------
    t: float
        time
    Omega: float
        frequency of sinc

    Returns
    -------
    float
        second integral of the sinc of frequency Omega between time 0 and time t
    """

    return (np.cos(Omega * t) + Omega * t * sici(Omega * t)[0]) / (Omega * np.pi)


def dirichlet_integral(t, period, n_components):
    """
    integral of dirichlet kernel

    Parameters
    ----------
    t: float
        time
    period: float
        period of the signal
    n_components: int
        number of components of dirichlet kernel

    Returns
    -------
    float
        integral of the dirichlet kernel of n_components components and period
        period between time 0 and time t
    """

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


def dirichlet_component_integral(t, period, component):
    """
    integral of one cosine of the expansion dirichlet kernel
    assuming D_n(t) = 1/2pi (1+2 sum_(k=1 to n) (cos kx)

    Parameters
    ----------
    t: float
        time
    period: float
        period of the signal
    component: int
        component of interest

    Returns
    -------
    float
        integral of the nth component of the dirichlet kernel of period
        period between time 0 and time t
    """

    if isinstance(component, int) and component == 0:
        integral = t
    else:
        integral = (
            1
            / (component * 2 * np.pi / period)
            * np.sin(component * 2 * np.pi / period * t)
        )
        if not isinstance(component, int):
            integral[np.where(component == 0)] = t

    return integral


def dirichlet_second_integral(t, period, n_components):
    """
    second integral of dirichlet kernel

    Parameters
    ----------
    t: float
        time
    period: float
        period of the signal
    n_components: int
        number of components of dirichlet kernel

    Returns
    -------
    float
        second integral of the dirichlet kernel of n_components components and period
        period between time 0 and time t
    """

    component_range = np.arange(1, n_components)
    integral = 1 / (2 * period) * t**2
    for n, component in enumerate(component_range):
        integral -= (
            2
            * (period / (2 * np.pi * component)) ** 2
            * np.cos(2 * np.pi / period * component * t)
        )
    return integral


def exp_int(exponent, t_start, t_end, tolerance=1e-18):
    """
    integral of complex exponential with exponent exponent

    Parameters
    ----------
    t: float
        time
    t_start: float
        start time of integration
    t_end: float
        end time of integration
    tolerance: float
        minimal absolute value of exponent

    Returns
    -------
    float
        integral of the complex exponential with exponent exponent
        between times t_start and t_end

    Raises
    ------
    ValueError
        If t_start and t_end do not have the same length
    """
    if len(t_start) != len(t_end):
        raise ValueError(
            "You should have as many end times as start times for the integrals of the exponentials"
        )
    integrals = np.zeros((len(exponent), len(t_start)), dtype=np.complex_)
    t_start = np.atleast_2d(t_start)
    t_end = np.atleast_2d(t_end)
    for n, exp_n in enumerate(exponent):
        if np.abs(exp_n) > tolerance:
            integrals[n, :] = (np.exp(exp_n * t_end) - np.exp(exp_n * t_start)) / exp_n
        else:
            integrals[n, :] = t_end - t_start
    return integrals


def indicator_matrix(dimensions, indices_list):
    """
    Parameters
    ----------
    dimensions: tuple
        tuple of ints, shape of the matrix to be created
    indices_list: list
        list of tuples, each of which specifies a location at which the matrix
        should be nonzero

    Returns
    -------
    np.ndarray
        indicator matrix with shape dimensions and with one as an entry at the
        locations in indices_list
    """

    mat = np.zeros(dimensions)
    for indices in indices_list:
        mat[indices] += 1

    return mat


def singular_value_projection_w_matrix(shape, sensing_matrix, b, rank, tol, lr):
    """
    Singular Value Projection algorithm taken from Jain, Meka and Dhillon (2010)

    Parameters
    ----------
    shape: tuple
        tuple of ints, shape of the desired matrix
    sensing_matrix: np.ndarray
        measurement matrix which is applied to flattened version of matrix of interest
    b: np.ndarray
        results from applying sensing_matrix to flattened version of matrix of interest
    rank: int
        rank of matrix of interest
    tol: float
        tolerance on the error of the recovery
    lr: float
        learning rate of iterative algorithm

    Returns
    -------
    np.ndarray
        low rank matrix which fits the measurments given, following the SVP algorithm
    """

    X = np.zeros(shape)
    # TODO check if this should stay here or not
    # i.e. run tests again for unknown case
    # X = np.reshape( np.linalg.lstsq(sensing_matrix,b, rcond = None)[0] , shape)
    n_iterations = 100000
    for i in range(n_iterations):
        error = sensing_matrix.dot(X.flatten()) - b.T
        if i > 0 and np.linalg.norm(error) < len(error) * tol:
            print("exited at iteration ", i)
            break
        Y = X - lr * np.reshape(sensing_matrix.T.dot(error.T), shape)

        Y[np.isnan(Y)] = 0
        Y[np.isinf(Y)] = 0
        # obtain the SVD and crop the singular values
        U, s, Vh = np.linalg.svd(Y, full_matrices=True)
        S = np.zeros(shape)
        S[0:rank, 0:rank] = np.diag(s[0:rank])

        X = U.dot(S).dot(Vh)
    return X
