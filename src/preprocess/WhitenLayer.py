# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:39:41 2021

@author: salva
"""

import numpy as np
import tensorflow as tf
from typing import Tuple

from utilities import GeneralUtilities


##############################################################################

class WhitenLayer(tf.keras.layers.Layer):
    """
    Layer that applies a spectral whitening to an input time series.
    """

    def __init__(
            self,
            sample_rate: float = 2048.,
            **kwargs
        ) -> None:
        """
        Function to initialize the object.
        
        Parameters
        ----------
        sample_rate : float, optional
            Sample rate of the signal [Hz]. Default value is 2048.
        """
        super(WhitenLayer, self).__init__(**kwargs)
        self.norm = np.sqrt(sample_rate)


    def build(
            self, 
            input_shape: Tuple[int, int]
        ) -> None:
        """
        Function to build the graph of the layer.

        Parameters
        ----------
        input_shape : Tuple[int, int]
            Shape of the input to the layer.
        """

        super(WhitenLayer, self).build(input_shape)


    def call(
            self, 
            data: tf.Tensor
        ) -> tf.Tensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_samples, n_detectors)
            A batch of input mono waveforms, n_detectors (n_channels) should 
            be last.

        Returns
        -------
        tf.Tensor, shape = (None, n_samples, n_detectors)
            The corresponding whitened signals.
        """
        data_ref = GeneralUtilities.broadcast_dim(data)

        withen = []
        for i in range(data.get_shape()[-1]):
            x = data_ref[..., i]
            x = tf.cast(x, tf.complex64)
    
            spec = tf.signal.fft(x)

            real = tf.cast(tf.math.real(spec), tf.complex64)
            imag = tf.cast(tf.math.imag(spec), tf.complex64) 
            spec_conj = real - imag * 1j
            psd = spec * spec_conj
            psd = tf.cast(tf.math.real(psd), tf.complex64)
    
            x = spec / tf.math.sqrt(psd)
            x = tf.signal.ifft(x)
            x = tf.cast(tf.math.real(x) * self.norm, self.dtype)
            x = tf.expand_dims(x, axis = -1)
            withen = x if (i == 0) else tf.concat([withen, x], axis = -1)

        return withen


##############################################################################