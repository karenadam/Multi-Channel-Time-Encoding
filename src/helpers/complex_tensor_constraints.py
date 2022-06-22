import numpy as np
import src
import src.helpers
from src import *



class complex_vector_constraints(object):
    """
    Can return constraints on a complex vector which
    is the concatenation of [real, imag] parts

    ATTRIBUTES
    ----------
    n_complex_elements: int
        number of complex elements the constraints should take into account
    """

    def __init__(self, n_complex_elements):
        """
        PARAMTERS
        ----------
        n_complex_elements: int
            number of complex elements the constraints should take into account
        """
        self.n_complex_elements = n_complex_elements

    def get_conjugate_symmetry_constraint(self, i_entry):
        """
        PARAMETERS
        ----------
        i_entry: int
            complex entry for which conjugate symmetry should be imposed

        RETURNS
        -------
        np.ndarray
            measurement vector which imposes conjugate symmetry on the
            component i_entry (and -i_entry) (the result of the application
            of which is 0)
        """
        equal_real_constraint = np.eye(
            1, 2 * self.n_complex_elements, i_entry
        ) - np.eye(
            1, 2 * self.n_complex_elements, self.n_complex_elements - 1 - i_entry
        )
        opposite_imag_constraint = np.eye(
            1, 2 * self.n_complex_elements, 2 * self.n_complex_elements - 1 - i_entry
        ) + np.eye(
            1,
            2 * self.n_complex_elements,
            self.n_complex_elements + i_entry,
        )
        return np.concatenate([equal_real_constraint, opposite_imag_constraint])

    def get_conjugate_symmetry_constraints(self):
        """
        RETURNS
        -------
        np.ndarray
            measurement matrix which imposes conjugate symmetry on the components
             of the vector (the result of the application of which is 0)
        """

        n_constraints = int(np.ceil(self.n_complex_elements / 2))
        return np.concatenate(
            [
                self.get_conjugate_symmetry_constraint(i_entry)
                for i_entry in range(n_constraints)
            ]
        )


class complex_tensor_constraints(object):
    """
    Can return constraints on a complex tensor where the constraints are
    in the form of measurement vectors applied to the concatenation of the
    flattened real parts and the flattened imaginary parts

    The application of every constraint vectors yields a result of zero

    ATTRIBUTES
    ----------
    shape: tuple
        shape of the complex tensor the constraints are applied to
    num_dimensions: int
        number of dimensions that the tensor has
    """

    def __init__(self, shape):
        """
        PARAMETERS
        ----------
        shape: tuple
            shape of the complex tensor the constraints are applied to
        """
        self.shape = shape
        self.num_dimensions = len(shape)

    def real_constraints(self, indices):
        """
        gets a vector which imposes a reality constraint on the element at indices by setting their
        imaginary part to zero

        PARAMETERS
        ----------
        indices: tuple
            tuple indicating the location of the element of the tensor which should be
            set to real

        RETURNS
        -------
        np.ndarray
            measurement vector which imposes reality on the element at indices by forcing
            its imaginary part to be zero
        """

        op_i = np.zeros(self.shape)
        op_i[indices] = 1
        flat_op_i = np.atleast_2d(
            np.concatenate(
                [op_i.flatten(), np.zeros((np.product(self.shape)))]
            )
        )
        return flat_op_i

    def equality_constraints(self, indices_1, indices_2):
        """
        gets a measurment matrix which imposes an equality constraint between the elements at indices
        indices_1 and indices_2 by setting their real parts and their imaginary parts to be respectively
        equal

        PARAMETERS
        ----------
        indices_1: tuple
            tuple indicating the location of the first element of the tensor
        indices_2: tuple
            tuple indicating the location of the second element of the tensor

        RETURNS
        -------
        np.ndarray
            measurement matrix which imposes equality on the elements at indices indices_1 and indices_2
            by forcing their imaginary parts and their real parts to be respectively equal
        """

        op_i = np.zeros(self.shape)
        op_i[indices_1] += 1
        op_i[indices_2] -= 1

        flat_real_op_i = np.atleast_2d(
            np.concatenate(
                [np.zeros((np.product(self.shape))), op_i.flatten()]
            )
        )
        flat_imag_op_i = np.atleast_2d(
            np.concatenate(
                [op_i.flatten(), np.zeros((np.product(self.shape)))]
            )
        )
        return np.concatenate([flat_real_op_i, flat_imag_op_i])

    def complex_conjugate_constraints(self, indices_1, indices_2):
        """
        gets a measurment matrix which imposes a complex conjugation constraint between the elements at indices
        indices_1 and indices_2 by setting their real parts  to be equal and their imaginary parts to be
        opposites of each other

        PARAMETERS
        ----------
        indices_1: tuple
            tuple indicating the location of the first element of the tensor
        indices_2: tuple
            tuple indicating the location of the second element of the tensor

        RETURNS
        -------
        np.ndarray
            measurement matrix which imposes complex conjugation on the elements at indices indices_1 and
            indices_2 by forcing their real parts  to be equal and their imaginary parts to be
            opposites of each other
        """
        imag_op_i = src.helpers.kernels.indicator_matrix(
            self.shape, [indices_1, indices_2]
        )

        real_op_i =src.helpers.kernels.indicator_matrix(
            self.shape, [indices_1]
        ) - src.helpers.kernels.indicator_matrix(self.shape, [indices_2])

        flat_real_op_i = np.atleast_2d(
            np.concatenate(
                [np.zeros((np.product(self.shape))), real_op_i.flatten()]
            )
        )
        flat_imag_op_i = np.atleast_2d(
            np.concatenate(
                [imag_op_i.flatten(), np.zeros((np.product(self.shape)))]
            )
        )

        return np.concatenate([flat_real_op_i, flat_imag_op_i])

    def real_center_coefficients_along_single_components(self):
        def index_values(dim):
            return (
                [
                    0,
                    int(self.shape[dim] / 2),
                    int(self.shape[dim] / 2) + 1,
                ]
                if self.shape[dim] % 2 == 0
                else [0]
            )

        indices = [index_values(dim) for dim in range(self.num_dimensions)]
        return np.concatenate(
            [
                self.real_constraints((dim_i, dim_j, dim_k))
                for dim_i in indices[0]
                for dim_j in indices[1]
                for dim_k in indices[2]
            ]
        )

    def complex_conjugate_symmetry_along_single_components(self):
        """
        gets a measurement matrix which enforces that the components are
        conjugate symmetric along one axis of the tensor

        RETURNS
        -------
        np.ndarray
            measurement matrix which enforces that the components are
            conjugate symmetric along one axis of the tensor
        """

        def get_constraint(component_index):
            def index_values(dim):
                return (
                    [
                        0,
                        int(self.shape[dim] / 2),
                        int(self.shape[dim] / 2) + 1,
                    ]
                    if self.shape[dim] % 2 == 0
                    else [0]
                )

            indices = [index_values(dim) for dim in range(self.num_dimensions)]
            indices[component_index] = np.arange(
                1, self.shape[component_index]
            ).tolist()
            index_multiplier = [1] * self.num_dimensions
            index_multiplier[component_index] = -1

            if len(indices[component_index]) == 0:
                return np.zeros((0, 2 * np.product(self.shape)))

            return np.concatenate(
                [
                    self.complex_conjugate_constraints(
                        (dim_i, dim_j, dim_k),
                        (
                            index_multiplier[0] * dim_i,
                            index_multiplier[1] * dim_j,
                            index_multiplier[2] * dim_k,
                        ),
                    )
                    for dim_i in indices[0]
                    for dim_j in indices[1]
                    for dim_k in indices[2]
                ]
            )

        return np.concatenate(
            [get_constraint(dim) for dim in range(self.num_dimensions)]
        )

    def center_point_reflection_complex_conjugates(self):
        """
        gets a measurement matrix which enforces that the components are
        conjugate symmetric around the center of the tensor

        RETURNS
        -------
        np.ndarray
            measurement matrix which enforces that the components are
            conjugate symmetric around the center of the tensor
        """

        return np.concatenate(
            [
                self.complex_conjugate_constraints(
                    (dim_i, dim_j, dim_k), (-dim_i, -dim_j, -dim_k)
                )
                for dim_i in range(self.shape[0])
                for dim_j in range(self.shape[1])
                for dim_k in range(self.shape[2])
            ]
        )

    def _get_equality_constraints(self, periods):
        def get_constraints(indices):
            ops = np.zeros((0, 2 * np.product(self.shape)))
            for n_component in range(self.num_dimensions):
                if periods[n_component] % 2 == 0 and indices[n_component] == int(
                    self.shape[n_component] / 2
                ):
                    equal_indices = list(indices)
                    equal_indices[n_component] = -indices[n_component]
                    ops = np.concatenate(
                        [ops, self.equality_constraints(indices, tuple(equal_indices))]
                    )
            return ops

        return np.concatenate(
            [
                get_constraints((dim_i, dim_j, dim_k))
                for dim_i in range(self.shape[0])
                for dim_j in range(self.shape[1])
                for dim_k in range(self.shape[2])
            ]
        )