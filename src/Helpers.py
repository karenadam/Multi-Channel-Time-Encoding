from src import *


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


def Di2(t, period, component):
    if component == 0:
        integral = t
    else:
        integral = (
            1
            / (component * 2 * np.pi / period)
            * np.sin(component * 2 * np.pi / period * t)
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


def singular_value_projection_w_matrix(shape, sensing_matrix, b, rank, tol, lr):

    X = np.zeros(shape)
    # TODO check if this should stay here or not
    # i.e. run tests again for unknown case
    # X = np.reshape( np.linalg.lstsq(sensing_matrix,b, rcond = None)[0] , shape)
    n_iterations = 100000
    for i in range(n_iterations):
        error = sensing_matrix.dot(X.flatten()) - b.T
        if i>0 and np.linalg.norm(error) < len(error)*tol:
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


# def singular_value_projection_w_matrix(shape, sensing_matrix, b, rank, tol, lr):
#     X = np.zeros(shape)
#     n_iterations = 100000
#     for i in range(n_iterations):
#         error = sensing_matrix.dot(X.flatten()) - b.T
#         Y = X - lr * np.reshape(sensing_matrix.T.dot(error.T), shape)
#         Y[np.isnan(Y)] = 0
#         Y[np.isinf(Y)] = 0
#         # obtain the SVD and crop the singular values
#         U, s, Vh = np.linalg.svd(Y, full_matrices=True)
#         S = np.zeros((U.shape[1], Vh.shape[0]))
#         S[0:rank, 0:rank] = np.diag(s[0:rank])
#         X = U.dot(S).dot(Vh)
#     return X
