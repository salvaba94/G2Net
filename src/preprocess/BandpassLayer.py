# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 00:01:09 2021

@author: salva
"""

import tensorflow as tf
import numpy as np
from scipy import signal
from scipy import interpolate
from functools import partial
from typing import Tuple, Mapping

from utilities import GeneralUtilities



##############################################################################


class BandpassLayer(tf.keras.layers.Layer):
    """
    Layer that applies a bandpass filter in the frequency domain, where the 
    possibility of training frequency response from the filter is given.
    """

    def __init__(
            self, 
            sample_rate: float = 2048.,
            degree: int = 8,
            f_band: Tuple[float, float] = (20, 500),
            n_samples: int = 4096,
            **kwargs
        ) -> None:
        """
        Funtion to initialize the lo

        Parameters
        ----------
        

        """
    
        super(BandpassLayer, self).__init__(**kwargs)

        if f_band[-1] <= f_band[0]:
            raise ValueError("Maximum frequency in spectral band should be \
                             higher than minimum frequency")

        f_nyq = sample_rate / 2.
        f_min = f_band[0] / f_nyq
        f_max = f_band[-1] / f_nyq
        self.norm = tf.constant(np.sqrt((f_band[-1] - f_band[0]) / f_nyq), 
                                dtype = self.dtype)

        f_fft = np.fft.rfftfreq(n_samples, d = 1./sample_rate)

        b, a = signal.butter(degree, (f_min, f_max), btype = "bandpass")
        w, gain = signal.freqz(b, a, worN = 2 * f_fft.shape[0])
        f = (f_nyq / np.pi) * w
        gain = np.abs(gain)

        gain_f = interpolate.interp1d(f, gain, fill_value = "extrapolate")
        f_response = gain_f(f_fft)
        f_response = f_response[np.newaxis, :]

        self.f_response = tf.Variable(initial_value = f_response, 
                                 trainable = self.trainable,
                                 name = self.name + "/f_response", 
                                 dtype = self.dtype)




    def build(
            self, 
            input_shape: Tuple[int, int]
        ) -> None:
        """
        Function to build the graph of the layer. Adds trainable and non-
        trainable parameters.

        Parameters
        ----------
        input_shape : Tuple[int, int]
            Shape of the input to the layer.
        """

        if self.trainable:
            self.trainable_weights.append(self.f_response)
        else:
            self.non_trainable_weights.append(self.f_response)
        super(BandpassLayer, self).build(input_shape)


    def call(
            self, 
            data: tf.Tensor
        ) -> tf.Tensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_samples, n_detectors)
            A batch of input waveforms, n_detectors (n_channels) should 
            be last.

        Returns
        -------
        tf.Tensor, shape = (None, n_samples, n_detectors)
            The corresponding batch of bandpassed signals.
        """

        data_ref = GeneralUtilities.broadcast_dim(data)
        bandpass = []
        for i in range(data_ref.get_shape()[-1]):
            x = data_ref[..., i]
            x = tf.cast(x, self.dtype)
            spec = tf.signal.rfft(x)
            spec *= tf.cast(self.f_response, tf.complex64)
            x = tf.signal.irfft(spec)
            x = tf.cast(x, self.dtype)
            x = tf.expand_dims(x, axis = -1)
            bandpass = x if (i == 0) else tf.concat([bandpass, x], axis = -1)
        
        bandpass /= self.norm
        return bandpass


    
##############################################################################