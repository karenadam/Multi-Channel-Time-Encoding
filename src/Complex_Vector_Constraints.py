from src import *


class complex_vector_constraints(object):
    """
    A class that can return constraints on a complex vector which
    is the concatenation of [real, imag] parts

    Attributes
    ----------
    n_complex_elements: int
        number of complex elements the constraints should take into account
    """

    def __init__(self, n_complex_elements):
        """
        Parameters
        ----------
        n_complex_elements: int
            number of complex elements the constraints should take into account
        """
        self.n_complex_elements = n_complex_elements

    def get_conjugate_symmetry_constraint(self, i_entry):
        """
        Parameters
        ----------
        i_entry: int
            complex entry for which conjugate symmetry should be imposed

        Returns
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
        Returns
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
    def __init__(self, shape_complex_tensor):
        self.shape_complex_tensor = shape_complex_tensor

    def real_constraints(self, indices):
        op_i = np.zeros(self.shape_complex_tensor)
        op_i[indices] = 1
        flat_op_i = np.atleast_2d(
            np.concatenate(
                [op_i.flatten(), np.zeros((np.product(self.shape_complex_tensor)))]
            )
        )
        return flat_op_i

    def equality_constraints(self, indices_1, indices_2):
        op_i = np.zeros(self.shape_complex_tensor)
        op_i[indices_1] += 1
        op_i[indices_2] -= 1

        flat_real_op_i = np.atleast_2d(
            np.concatenate(
                [np.zeros((np.product(self.shape_complex_tensor))), op_i.flatten()]
            )
        )
        flat_imag_op_i = np.atleast_2d(
            np.concatenate(
                [op_i.flatten(), np.zeros((np.product(self.shape_complex_tensor)))]
            )
        )
        return np.concatenate([flat_real_op_i, flat_imag_op_i])

    def complex_conjugate_constraints(self, indices_1, indices_2):
        imag_op_i = Helpers.indicator_matrix(
            self.shape_complex_tensor, [indices_1, indices_2]
        )

        real_op_i = Helpers.indicator_matrix(
            self.shape_complex_tensor, [indices_1]
        ) - Helpers.indicator_matrix(self.shape_complex_tensor, [indices_2])

        flat_real_op_i = np.atleast_2d(
            np.concatenate(
                [np.zeros((np.product(self.shape_complex_tensor))), real_op_i.flatten()]
            )
        )
        flat_imag_op_i = np.atleast_2d(
            np.concatenate(
                [imag_op_i.flatten(), np.zeros((np.product(self.shape_complex_tensor)))]
            )
        )

        return np.concatenate([flat_real_op_i, flat_imag_op_i])
