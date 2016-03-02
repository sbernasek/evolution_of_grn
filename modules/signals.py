__author__ = 'Sebi'

import numpy as np
import random as rd


class Signal:
    """
    Defines an input or output signal
    """

    def __init__(self, name, duration=10, dt=0.001, signal=None, channels=None):
        self.name = name
        self.duration = 0
        self.signal = signal
        self.dt = None

        if channels is None:
            self.channels = channels
        else:
            self.channels = list(range(0, channels))

        self.time = np.arange(0, duration, dt)
        self.update_duration()
        self.dt = self.time[1] - self.time[0]
        # self.signal = np.zeros((len(self.channels), len(self.time)))
        self.signal = signal

    def update_duration(self):
        """
        Computes signal duration.
        """
        duration = self.time[-1] - self.time[0]
        if duration < 0:
            print('Signal time vector is non-chronological')
            self.duration = 0
        else:
            self.duration = duration

    def add_noise(self, mu, sigma):
        """
        Adds gaussian white noise to all signal channels.

        Parameters:
            mu - mean value of noise
            sigma - standard deviation of noise
        """
        for row, channel in enumerate(self.signal):
            self.signal[row, :] = np.asarray([data_point + rd.normalvariate(mu, sigma) for data_point in channel])

    def generate_signal(self, f, channels):
        """
        Generates signal vector given a function f.

        Parameters:
            f (fnc) - function of time describing signal magnitude
            channels (list) - indices of signal channels on which signal is generated
        Returns:
            self.signal (np array) - vector describing signal magnitude
        """



        if self.time is None:
            print('Signal\'s time vector has not been defined.')
        else:
            self.signal = np.zeros((len(self.channels), len(self.time)))
            for channel in channels:
                self.signal[channel, :] = np.asarray([f(time_point) for time_point in self.time])
            self.update_duration()

    def merge_signals(self, second_signal, shift=False, gap=None):
        """
        Merges two signals by horizontally stacking their time and magnitude vectors.

        Parameters:
            second_signal (signal object) -  signal to succeed first signal
            shift (bool) - if true, second signal is shifted in time such that it follows the first signal
            gap (float) - time between end of first and start of second signal
        Returns:
            self.time (np array) - updated signal time vector
            self.signal (np array) - updated signal magnitude vector
        """
        if gap is None:
            gap = 0

        if len(self.signal[:, 0]) != len(second_signal.signal[:, 0]):
            print('Signals to be merged have a different number of channels. Merge aborted.')
            return self

        if shift is False:
            self.time = np.hstack((self.time, second_signal.time))
        else:
            shifted_times = np.asarray([gap + self.time[-1] + time_increment for time_increment in second_signal.time])
            self.time = np.hstack((self.time, shifted_times))

        self.signal = np.hstack((self.signal, second_signal.signal))
        self.update_duration()
        return self

    def step(self, magnitude=1, channels=None, return_signal=False):
        """
        Creates constant signal.
           e.g. magnitude = 0 is a null signal, magnitude = 1 is a unit step.

        Parameters:
            magnitude (float) - signal magnitude
            return_signal (bool) - if true, returns signal object
            channels (list) - list of channels to which step is assigned
        """

        if channels is None:
            channels = list(range(0, len(self.channels)))

        self.generate_signal(lambda x: magnitude, channels)

        if return_signal is True:
            return self

    def pulse(self, magnitude=1, off_duration=None, channels=None, return_signal=False):
        """
        Creates temporary step input with long "off" state.

        Parameters:
            magnitude (float) - size of pulse
            off_duration (float) - duration of null signal
            channels (list) - list of channels to which step is assigned
            return_signal (bool) - if true, returns signal object
        """

        temp_off_signal = Signal('off', duration=off_duration, dt=self.dt, channels=len(self.channels))
        temp_off_signal.step(0, channels, return_signal=True)

        self.step(magnitude, channels)
        self.merge_signals(temp_off_signal, shift=True, gap=self.dt)

        if return_signal is True:
            return self


