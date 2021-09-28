

class Network(object):
    def __init__(self, layers : list = None):
        self.layers = []
        if layers is not None:
            for l in len(layers):
                if l>0:
                    assert layers.num_inputs == self.layers[-1].num_outputs
                self.layers = layers[i]