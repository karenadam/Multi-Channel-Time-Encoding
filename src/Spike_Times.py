from src import *


class spikeTimes(object):
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.spikes = [[] for _ in range(n_channels)]

    def add(self, channel, item):
        if isinstance(item, (list)):
            self.spikes[channel].extend(item)
        else:
            self.spikes[channel].append(item)

    def get_spikes_of(self, channel, asSpikeTimesObject=False):
        if asSpikeTimesObject:
            spikes_single = spikeTimes(n_channels=1)
            spikes_single.spikes = [self.spikes[channel]]
            return spikes_single
        return np.array(self.spikes[channel])

    def get_total_num_spikes(self):
        total_num = 0
        for ch in range(self.n_channels):
            total_num += len(self.spikes[ch])
        return total_num

    def get_total_num_spike_diffs(self):
        total_num = 0
        for ch in range(self.n_channels):
            if len(self.spikes[ch]) > 0:
                total_num += len(self.spikes[ch]) - 1
        return total_num

    def get_n_spikes_of(self, channel):
        return len(self.spikes[channel])

    def get_total_num_constraints(self):
        total_num = 0
        for ch in range(self.n_channels):
            total_num += len(self.spikes[ch]) - 1
        return total_num

    def get_const_num_constraints(self, rMax):
        total_num = 0
        for ch in range(self.n_channels):
            total_num += min(len(self.spikes[ch]) - 1, rMax)
        return total_num

    def get_midpoints(self, channel):
        return (self.get_spikes_of(channel)[1:] + self.get_spikes_of(channel)[:-1]) / 2

    def get_all_midpoints(self):
        midpoints = np.zeros((self.get_total_num_constraints(), 1))
        start_ind = 0
        for ch in range(self.n_channels):
            midpoints_ch = self.get_midpoints(ch)
            midpoints[start_ind : start_ind + len(midpoints_ch)] = np.atleast_2d(
                midpoints_ch
            ).T
            start_ind += len(midpoints_ch)
        return (midpoints.T[0]).tolist()

    def print(self):
        for ch in range(self.n_channels):
            print("\nSpikes of Channel {}".format(ch))
            for s in self.get_spikes_of(ch):
                print(s, end=", ")

    def corrupt_with_gaussian(self, snr):
        corrupted_spikeTimes = spikeTimes(self.n_channels)
        for ch in range(self.n_channels):
            spike_dif = self.get_spikes_of(ch)[1:] - self.get_spikes_of(ch)[:-1]
            spike_dif_power = np.linalg.norm(spike_dif) / len(spike_dif)
            noise_power = snr * spike_dif_power
            added_noise = np.random.normal(
                0, noise_power, size=self.get_spikes_of(ch).shape
            ).tolist()
            new_spikes_for_ch = [
                self.spikes[ch][l] + added_noise[l] for l in range(len(added_noise))
            ]
            corrupted_spikeTimes.add(ch, new_spikes_for_ch)
        return corrupted_spikeTimes
