import numpy as np
from src import *


class Network(object):
    """
    A class used to represent a spiking neural network

    Attributes
    ----------
    layers: list
        a list of layer objects that make up the neural network
    """

    def __init__(self, layers: list = None):
        """
        Parameters
        ----------
        layers: list
            a list of layer objects that make up the neural network

        Raises
        ------
        ValueError
            If consecutive layers have inconsistent numbers of inputs and outputs
        """

        self.layers = []
        if layers is not None:
            for i in range(len(layers)):
                if i > 0 and layers[i].num_inputs != self.layers[i - 1].num_outputs:
                    raise ValueError(
                        "The number of inputs of layer "
                        + str(i)
                        + " is different than the number of outputs of layer "
                        + str(i - 1)
                    )
                self.layers.append(layers[i])

    def __repr__(self):
        if len(self.layers) == 0:
            return "Network with no layers"
        if len(self.layers) == 1:
            return (
                "Network with one layer, "
                + str(self.layers[0].num_inputs)
                + " inputs and "
                + str(self.layers[0].num_outputs)
                + " outputs"
            )
        return (
            "Network with "
            + str(len(self.layers))
            + " layers, with "
            + str(self.layers[0].num_inputs)
            + " inputs, "
            + str([self.layers[i].num_outputs for i in range(len(self.layers) - 1)])
            + " hidden nodes, and "
            + str(self.layers[-1].num_outputs)
            + " outputs"
        )
