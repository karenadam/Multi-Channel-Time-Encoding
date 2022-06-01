import sys
import os
import numpy as np

sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0] + "/..")
from src import *


class TestCanCreateNetwork:
    def test_can_create_empty_network(self):
        net = Network([])

    def test_can_create_single_layer_network(self):
        layer = Layer(num_inputs = 3, num_outputs = 4)
        net = Network([layer])

    def test_can_create_two_layer_network(self):
        layer_1 = Layer(num_inputs = 3, num_outputs = 4)
        layer_2 = Layer(num_inputs = 4, num_outputs = 2)
        net = Network([layer_1, layer_2])

    def test_throws_error_if_wrong_layer_type(self):
        try:
            net = Network([4,2],[2,3])
        except:
            return
        assert False

    def test_throws_error_if_layer_mismatch(self):
        layer_1 = Layer(num_inputs = 3, num_outputs = 4)
        layer_2 = Layer(num_inputs = 3, num_outputs = 2)
        try:
            net = Network([layer_1, layer_2])
        except:
            return
        assert False
