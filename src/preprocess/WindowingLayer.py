# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 09:40:10 2021

@author: salva
"""

import tensorflow as tf
import numpy as np
from scipy import signal
from typing import Union, Tuple, Mapping

from utilities import GeneralUtilities



##############################################################################

class WindowingLayer(tf.keras.layers.Layer):
    """
    Layer that applies a window function to an input time series.
    """

    def __init__(
            self, 
            window: Union[str, Tuple[str, float]] = ("tukey", 0.1),
            window_len: int = 4096,
            **kwargs
        ) -> None:
        """

        Parameters
        ----------
        window : str or Tuple[str, float], optional
            The type of window to create with any parameter it might need. 
            The default is "tukey" with alpha 0.1.
        window_len : int, optional
            The number of samples in the window (set it to the signal length). 
            The default is 4096.

        Returns
        -------
        None
        """
    
        super(WindowingLayer, self).__init__(**kwargs)

        self.window = window
        self.window_len = window_len

        sig = signal.get_window(window, window_len)[np.newaxis, :, np.newaxis]
        self.window = tf.Variable(initial_value = sig, trainable = self.trainable,
                                  name = self.name + "/window", dtype = self.dtype)


    def build(
            self, 
            input_shape: Tuple[int, int]
        ) -> None:
        if self.trainable:
            self.trainable_weights.append(self.window)
        else:
            self.non_trainable_weights.append(self.window)
        super(WindowingLayer, self).build(input_shape)


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
            The corresponding batch of windowed waveforms.
        """

        x = GeneralUtilities.broadcast_dim(data)
        x = tf.cast(x, self.dtype)
        x *= self.window
        return x


    def get_config(
            self
        ) -> Mapping[str, float]:
        """
        Function to get the configuration parameters of the object.
        
        Returns
        -------
        Mapping[str, float]
            Dictionary containing the configuration parameters of the object.
        """
        config = {
            "window" : self.window,
            "window_len" : self.window_len
        }
        
        config.update(super(WindowingLayer, self).get_config())
        return config


##############################################################################