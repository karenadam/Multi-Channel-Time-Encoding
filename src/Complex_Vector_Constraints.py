from src import *

# class that can return constraints on a complex vector which is the concatenation of [real, imag] parts
class complex_vector_constraints(object):
    def __init__(self, n_complex_elements):
        self.n_complex_elements = n_complex_elements

    def get_complex_conjugate_constraint(self, i_entry):
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
        n_constraints = int(np.ceil(self.n_complex_elements / 2))
        return np.concatenate(
            [
                self.get_complex_conjugate_constraint(i_entry)
                for i_entry in range(n_constraints)
            ]
        )
