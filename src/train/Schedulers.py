# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 19:03:19 2021

@author: salva
"""
import tensorflow as tf


##############################################################################

class CosineAnnealingRestarts(tf.keras.experimental.CosineDecayRestarts):
    """
    This class inherits from CosineDecayRestarts. It adds the functionality 
    of returning the initial learning rate when a float conversion is called 
    and the assignment functionality.
    """

    def __init__(
            self,
            dtype: type = tf.float32,
            **kwargs
        ) -> None:
        """
        Object initialization function.

        Parameters
        ----------
        dtype : type
            Data type of the learning rate.
        """

        super(CosineAnnealingRestarts, self).__init__(**kwargs)
        self.dtype = dtype


    def __float__(
            self
        ) -> float:
        """
        Magic method for float() calls.

        Returns
        -------
        float
            Initial learning rate.
        """

        return self.initial_learning_rate
    

    def assign(
            self,
            learning_rate: float
        ) -> float:
        """
        Method to set the value of the initial learning rate.

        Parameters
        -------
        learning_rate : float
            Initial learning rate to set.
        """

        self.initial_learning_rate = float(learning_rate)


##############################################################################