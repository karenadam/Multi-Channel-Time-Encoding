import numpy as np


class spikeTimes(object):
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.spikes = [[] for _ in range(n_channels)]

    def add(self, channel, item):
        if isinstance(item, (list)):
            self.spikes[channel].extend(item)
        else:
            self.spikes[channel].append(item)

    def get_spikes_of(self, channel):
        return np.array(self.spikes[channel])

    def get_total_num_spikes(self):
        total_num = 0
        for ch in range(self.n_channels):
            total_num += len(self.spikes[ch])
        return total_num

    def get_n_spikes_of(self, channel):
        return len(self.spikes[channel])

    def get_midpoints(self, channel):
        return (self.get_spikes_of(channel)[1:] + self.get_spikes_of(channel)[:-1]) / 2
