# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:44:02 2021

@author: salva
"""

import tensorflow as tf
import numpy as np
from functools import partial
from typing import Tuple

from utilities import GeneralUtilities


##############################################################################

class TukeyWinLayer(tf.keras.layers.Layer):
    """
    Layer that applies a Tukey window function to an input time series, where 
    the possibility of training the shape parameter is given.
    """

    def __init__(
            self, 
            initial_alpha: float = 0.25,
            **kwargs
        ) -> None:
        """

        Parameters
        ----------
        initial_alpha : float, optional
            Shape parameter of the tukey window. The default is 0.25.
        """
    
        super(TukeyWinLayer, self).__init__(**kwargs)

        self.alpha = tf.Variable(initial_value = initial_alpha, 
                                 trainable = self.trainable,
                                 name = self.name + "/alpha", 
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
            self.trainable_weights.append(self.alpha)
        else:
            self.non_trainable_weights.append(self.alpha)
        super(TukeyWinLayer, self).build(input_shape)


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
            The corresponding batch of windowed signals.
        """

        x = GeneralUtilities.broadcast_dim(data)
        x = tf.cast(x, self.dtype)
        w_len = tf.shape(x)[1]
        window = GeneralUtilities.broadcast_dim(self._get_window(w_len))
        x *= window
        return x


    def _get_ones(
            self,
            w_len: int
        ) -> tf.Tensor:
        """
        Case for a null shape parameter.

        Parameters
        ----------
        w_len : int
            Length of the window.

        Returns
        -------
        tf.Tensor : shape = (window_len,)
            A window of all ones.
        """
 
        window = tf.ones(w_len, dtype = self.dtype)
        return window

        
    def _get_hann(
            self,
            w_len: int
        ) -> tf.Tensor:
        """
        Case for a unity shape parameter.

        Parameters
        ----------
        w_len : int
            Length of the window.

        Returns
        -------
        tf.Tensor : shape = (window_len,)
            A hann window.
        """
 
        window = tf.signal.hann_window(w_len, periodic = False)
        window = tf.cast(window, dtype = self.dtype)
        return window


    def _get_tukey(
            self,
            w_len: int
        ) -> tf.Tensor:
        """
        Case for a non-null and non-unity shape parameter.

        Parameters
        ----------
        w_len : int
            Length of the window.

        Returns
        -------
        tf.Tensor : shape = (window_len,)
            A tukey window.
        """
 
        w_len_f = tf.cast(w_len, self.dtype)
       
        n = tf.range(0, w_len)
        width = tf.math.floor(self.alpha * (w_len_f - 1.)/2.)
        width = tf.minimum(width, w_len_f)
        width = tf.maximum(width, 0.)
        width = tf.cast(width, dtype = tf.int32)

        n_1 = tf.cast(n[:width + 1], dtype = self.dtype)
        n_2 = tf.cast(n[width + 1: w_len - width - 1], dtype = self.dtype)
        n_3 = tf.cast(n[w_len - width - 1:], dtype = self.dtype)
    
        window_1 = 0.5 * (1. + tf.math.cos(np.pi * (
            -1. + 2. * n_1 / self.alpha / (w_len_f - 1.))))
        window_2 = tf.ones(tf.shape(n_2))
        window_3 = 0.5 * (1. + tf.math.cos(np.pi * (
            - 2./self.alpha + 1. + 2. * n_3 / self.alpha / (w_len_f - 1.))))
    
        window = tf.concat((window_1, window_2, window_3), axis = 0)
        return window
    
    
    def _get_window(
            self,
            w_len: int
        ) -> tf.Tensor:
        """
        Tukey window getter.

        Parameters
        ----------
        w_len : int
            Length of the window.

        Returns
        -------
        tf.Tensor : shape = (window_len,)
            A tukey window handling the cases of null or unity shape parameters.
        """
        get_ones = partial(self._get_ones, w_len = w_len + 1)
        get_hann = partial(self._get_hann, w_len = w_len + 1)
        get_tukey = partial(self._get_tukey, w_len = w_len + 1)

        window = tf.case([(tf.less_equal(self.alpha, 0.), get_ones),
                          (tf.greater_equal(self.alpha, 1.), get_hann)], 
                         default = get_tukey)
        window = window[:-1]
        return window
    
    
##############################################################################