import numpy as np


class SpikeTimes(object):
    """
    A class used to represent spike times of one or multiple TEMs

    Attributes
    ----------
    n_channels: int
        number of channels for which spike times are stored
    spikes: list
        list of n_channels lists of floats that represent the time
        of the spikes
    """

    def __init__(self, n_channels):
        """
        Parameters
        ----------
        n_channels: number of channels for which spike times will be stored
        """

        self.n_channels = n_channels
        self.spikes = [[] for _ in range(n_channels)]

    def __repr__(self):
        return (
            "SpikeTimes with "
            + str(self.n_channels)
            + " channels and the following spike output: "
            + str(self.spikes)
        )

    def __str__(self):
        string = ""
        for ch in range(self.n_channels):
            string += "Spikes of Channel {}".format(ch) + "\n"
            for s in self.get_spikes_of(ch):
                string += str(s) + ", "
            string += "\n"
        return string

    def __getitem__(self, key):
        return self.get_spikes_of(key, asSpikeTimesObject=False)

    def add(self, channel, spikes):
        """
        Adds spike times to list of desired channel

        Parameters
        ----------
        channel: int
            channel whose spike times are being added to
        spikes: float or list
            float or list of floats which is/are added to the existing list of
            spike times for channel channel
        """

        if isinstance(spikes, (list)):
            self.spikes[channel].extend(spikes)
        else:
            self.spikes[channel].append(spikes)

    def get_spikes_of(self, channel, asSpikeTimesObject=False):
        """
        Parameters
        ----------
        channel: int
            desired channel
        asSpikeTimesObject: bool
            specifies if output should be in the form of a SpikeTimes object (if True)
            or in the form of a list (if False)

        Returns
        -------
        list or SpikeTimes object
            a list or a SpikeTimes object which contains the spike times of channel channel
        """

        if asSpikeTimesObject:
            spikes_single = SpikeTimes(n_channels=1)
            spikes_single.spikes = [self.spikes[channel]]
            return spikes_single
        return np.array(self.spikes[channel])

    def get_spikes(self):
        """
        Returns
        -------
        list
            list of lists of floats representing the spike times of all channels
        """

        return self.spikes

    def get_total_num_spikes(self):
        """
        Returns
        -------
        int
            total number of spikes emitted by all channels
        """

        total_num = 0
        for ch in range(self.n_channels):
            total_num += len(self.spikes[ch])
        return total_num

    def get_total_num_spike_diffs(self):
        """
        Returns
        -------
        int
            total number of consecutive spike pairs emitted by all channels
        """

        total_num = 0
        for ch in range(self.n_channels):
            if len(self.spikes[ch]) > 0:
                total_num += len(self.spikes[ch]) - 1
        return total_num

    def get_n_spikes_of(self, channel):
        """
        Parameters
        ----------
        channel: int
            index of channel of interest

        Returns
        -------
        int
            number of spikes emitted by that channel
        """

        return len(self.spikes[channel])

    def get_total_num_constraints(self):
        warnings.warn(
            "get_total_num_constraints is deprecated and will soon be removed."
            + "Please use get_total_num_spike_diffs instead.",
            Warning.DeprecationWarning,
        )
        return total_num_spike_diffs

    def get_const_num_constraints(self, rMax):
        """
        Parameters
        ----------
        rMax: int
            maximal rate of useful information per channel

        Returns
        -------
        int
            total number of consecutive spike pairs emitted by all channels,
            constrained by a maximal rate of information rMax per channel
        """

        total_num = 0
        for ch in range(self.n_channels):
            if len(self.spikes[ch]) > 0:
                total_num += min(len(self.spikes[ch]) - 1, rMax)
        return total_num

    def get_midpoints(self, channel):
        """
        Parameters
        ----------
        channel: int
            index of channel of interest

        Returns
        -------
        list
            list of floats which are the midpoints of consecutive spikes
            of channel channel
        """

        return (self.get_spikes_of(channel)[1:] + self.get_spikes_of(channel)[:-1]) / 2

    def get_all_midpoints(self):
        """
        Returns
        -------
        list
            list of list of floats which are the midpoints of consecutive spikes
            of each of the channels
        """

        midpoints = np.zeros((self.get_total_num_constraints(), 1))
        start_ind = 0
        for ch in range(self.n_channels):
            midpoints_ch = self.get_midpoints(ch)
            midpoints[start_ind : start_ind + len(midpoints_ch)] = np.atleast_2d(
                midpoints_ch
            ).T
            start_ind += len(midpoints_ch)
        return (midpoints.T[0]).tolist()

    def corrupt_with_gaussian(self, snr):
        """
        Parameters
        ----------
        snr: float
            signal-to-noise ratio of added gaussian noise

        Returns
        -------
        SpikeTimes
            SpikeTimes object which is similar to self but where the
            spike times are corrupted by Gaussian noise of the given SNR
        """

        corrupted_SpikeTimes = SpikeTimes(self.n_channels)
        for ch in range(self.n_channels):
            spike_dif = np.diff(self.get_spikes_of(ch))
            spike_dif_power = np.linalg.norm(spike_dif) / len(spike_dif)
            noise_power = snr * spike_dif_power
            added_noise = np.random.normal(
                0, noise_power, size=self.get_spikes_of(ch).shape
            ).tolist()
            new_spikes_for_ch = [
                self.spikes[ch][l] + added_noise[l] for l in range(len(added_noise))
            ]
            corrupted_SpikeTimes.add(ch, new_spikes_for_ch)
        return corrupted_SpikeTimes
