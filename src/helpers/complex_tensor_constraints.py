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
        self.num_dimensions = len(shape_complex_tensor)

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
        imag_op_i = helpers.kernels.indicator_matrix(
            self.shape_complex_tensor, [indices_1, indices_2]
        )

        real_op_i = helpers.kernels.indicator_matrix(
            self.shape_complex_tensor, [indices_1]
        ) - helpers.kernels.indicator_matrix(self.shape_complex_tensor, [indices_2])

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

    def real_center_coefficients_along_single_components(self):
        def index_values(dim):
            return (
                [
                    0,
                    int(self.shape_complex_tensor[dim] / 2),
                    int(self.shape_complex_tensor[dim] / 2) + 1,
                ]
                if self.shape_complex_tensor[dim] % 2 == 0
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
        def get_constraint(component_index):
            def index_values(dim):
                return (
                    [
                        0,
                        int(self.shape_complex_tensor[dim] / 2),
                        int(self.shape_complex_tensor[dim] / 2) + 1,
                    ]
                    if self.shape_complex_tensor[dim] % 2 == 0
                    else [0]
                )

            indices = [index_values(dim) for dim in range(self.num_dimensions)]
            indices[component_index] = np.arange(
                1, self.shape_complex_tensor[component_index]
            ).tolist()
            index_multiplier = [1] * self.num_dimensions
            index_multiplier[component_index] = -1

            if len(indices[component_index]) == 0:
                return np.zeros((0, 2 * np.product(self.shape_complex_tensor)))

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

        return np.concatenate(
            [
                self.complex_conjugate_constraints(
                    (dim_i, dim_j, dim_k), (-dim_i, -dim_j, -dim_k)
                )
                for dim_i in range(self.shape_complex_tensor[0])
                for dim_j in range(self.shape_complex_tensor[1])
                for dim_k in range(self.shape_complex_tensor[2])
            ]
        )

    def _get_equality_constraints(self, periods):
        def get_constraints(indices):
            ops = np.zeros((0, 2 * np.product(self.shape_complex_tensor)))
            for n_component in range(self.num_dimensions):
                if periods[n_component] % 2 == 0 and indices[n_component] == int(
                    self.shape_complex_tensor[n_component] / 2
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
                for dim_i in range(self.shape_complex_tensor[0])
                for dim_j in range(self.shape_complex_tensor[1])
                for dim_k in range(self.shape_complex_tensor[2])
            ]
        )