# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 20:09:00 2021

@author: salva
"""

import tensorflow as tf


##############################################################################

class RocLoss(tf.keras.losses.Loss):
    """
    Class implementing an AUC differentiable score approximation by the 
    Wilcoxon-Mann-Whitney Statistic.
    
    See "Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the 
    Wilcoxon-Mann-Whitney Statistic."
    """

    def __init__(
            self,
            gamma: float = 0.2,
            p: float = 2,
            **kwargs
        ) -> None:
        """
        Function to initialise the object.
        
        Parameters
        ----------
        gamma : float, optional
            Margin for the positive to negative difference. The default is 0.2
        p : float, optional
            Exponent to make the loss differentiable
        """
        
        tf.config.experimental_run_functions_eagerly(True)

        super(RocLoss, self).__init__(**kwargs)        
        self.gamma = gamma
        self.p = p


    def call(
            self, 
            y_true: tf.Tensor, 
            y_pred: tf.Tensor,
        ) -> tf.Tensor:
        """
        Forward pass of the loss function.
        
        Parameters
        ----------
        y_true : tf.Tensor, shape = (None,)
            Batch of ground truths.
        y_pred : tf.Tensor, shape = (None,)
            Batch of predictions.
            
        
        Returns
        -------
        tf.Tensor, shape = (None, n_time, n_freq, n_detectors)
            Value of the loss for the batch.
        """
        
        positive = tf.expand_dims(tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool)))
        negative = tf.expand_dims(tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool)))

        diff = tf.zeros_like(positive * negative) + positive - negative - self.gamma
        masked = tf.boolean_mask(diff, diff < 0.)
        return tf.reduce_sum((-masked) ** self.p)


##############################################################################
