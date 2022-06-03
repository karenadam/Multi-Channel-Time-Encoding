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
