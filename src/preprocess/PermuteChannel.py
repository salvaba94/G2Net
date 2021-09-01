# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 10:02:11 2021

@author: salva
"""

import tensorflow as tf
from functools import partial
from typing import Tuple, Mapping

from utilities import GeneralUtilities



##############################################################################

class PermuteChannel(tf.keras.layers.Layer):
    """
    Layer that randomly permutes the channels from data to avoid overfitting 
    in the context of G2Net.
    """

    def __init__(
            self, 
            rate: float = 0.5,
            **kwargs
        ) -> None:
        """
        Function to initialize the object.

        Parameters
        ----------
        rate : float, optional
            Probability of performing a permutation. The default 0.5.
        """
    
        super(PermuteChannel, self).__init__(**kwargs)
        self.rate = rate


    def build(
            self, 
            input_shape: Tuple[int, int]
        ) -> None:
        """
        Function to build the graph of the layer. Adds trainable and non-
        trainable parameters if any.

        Parameters
        ----------
        input_shape : Tuple[int, int]
            Shape of the input to the layer.
        """

        super(PermuteChannel, self).build(input_shape)



    def call(
            self, 
            data: tf.Tensor
        ) -> tf.Tensor:
        """
        Forward pass of the layer (requires channels to be last).

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_time, n_freq, n_detectors)
            A batch of channeled inputs, n_detectors (n_channels) should be last.

        Returns
        -------
        tf.Tensor, shape = (None, n_time, n_freq, n_detectors)
            The corresponding batch of channeled inputs.
        """

        x = GeneralUtilities.broadcast_dim(data)
        x = tf.cast(x, self.dtype)
        x = tf.cond(tf.random.uniform(0) < self.p, 
                    partial(self._permute_channels, data = x),
                    lambda x: x)
        return x


    @staticmethod
    def _permute_channels(
            data: tf.Tensor
        ) -> tf.Tensor:
        """
        This funtion applies a random permutations of the channels dimension 
        assuming channels last format.

        Parameters
        ----------
        data : tf.Tensor
            Input data.

        Returns
        -------
        tf.Tensor
            Permuted output data.
        """
    
        perm = tf.range(data.get_shape()[-1])
        perm = tf.random.shuffle()
        return tf.gather(data, perm, axis = -1)


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
            "rate" : self.rate
        }
        
        config.update(super(PermuteChannel, self).get_config())
        return config


##############################################################################