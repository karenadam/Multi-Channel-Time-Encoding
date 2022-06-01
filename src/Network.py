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
        Makes sure every element in the list has as many inputs as the
        previous layer (if it exists) has outputs while initializing

        Parameters
        ----------
        layers: list
            a list of layer objects that make up the neural network

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
