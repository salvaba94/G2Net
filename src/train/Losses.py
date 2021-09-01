# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 20:09:00 2021

@author: salva
"""

import tensorflow as tf
import numpy as np
# import torch
from typing import Tuple


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
        Function to initialize the object.
        
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
        
        positive = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        negative = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))
        positive = tf.expand_dims(positive, axis = 0)
        negative = tf.expand_dims(negative, axis = 1)

        diff = tf.zeros_like(positive * negative) + positive - negative - self.gamma
        masked = tf.boolean_mask(diff, diff < 0.)
        return tf.reduce_sum(tf.pow(-masked, self.p))



##############################################################################


class RocStarLoss(tf.keras.losses.Loss):
    """
    Class implementing a modified AUC differentiable score approximation by the 
    Wilcoxon-Mann-Whitney Statistic. 
    
    See https://github.com/iridiumblue/roc-star
    """

    def __init__(
            self,
            delta: float = 1.,
            sample_size: int = 1000,
            sample_size_g: int = 1000,
            update_rate_g: int = 10,
            dtype: type = tf.float32,
            **kwargs
        ):
        
        tf.config.experimental_run_functions_eagerly(True)

        super(RocStarLoss, self).__init__(**kwargs)        
        self.delta = delta
        self.sample_size = sample_size
        self.sample_size_g = sample_size_g
        self.update_rate_g = update_rate_g
        
        self.gamma = 0.
        self.steps = 0
        
        self.dtype = dtype
        
        size = np.maximum(sample_size, sample_size_g)
        
        self.y_pred_h = tf.convert_to_tensor(np.random.random((size, 1)), 
                                             dtype = self.dtype)
        self.y_true_h = tf.convert_to_tensor(np.random.randint(0, 2, (size, 1)),
                                             dtype = self.dtype)

    def call(
            self, 
            y_true: tf.Tensor, 
            y_pred: tf.Tensor,
        ) -> tf.Tensor:
        
        y_pred_c = tf.cast(y_pred, dtype = self.dtype)
        y_true_c = tf.cast(y_true, dtype = self.dtype)
        batch_size = tf.shape(y_pred_c)[0]
    
        if self.steps % self.update_rate_g == 0:
            self._update_gamma()
        self.steps += 1
        
        positive, negative = self._get_posneg(y = (y_pred, y_true))
        positive_h, negative_h = self._get_posneg()
            
        loss = self._compute_loss(positive, negative, positive_h, negative_h)
        
        self.y_pred_h = tf.concat((self.y_pred_h[batch_size:, :], 
                                   tf.stop_gradient(y_pred_c)), axis = 0)
        self.y_true_h = tf.concat((self.y_true_h[batch_size:, :], 
                                   tf.stop_gradient(y_true_c)), axis = 0)

        return loss


    def _get_posneg(
            self,
            y: Tuple[tf.Tensor, tf.Tensor] = None,
            use_g: bool = False
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        
        if y is not None:
            y_pred = y[0]
            y_true = y[-1]
        else:
            sample_size = self.sample_size_g if use_g else self.sample_size
            y_pred = self.y_pred_h[-sample_size:, :]
            y_true = self.y_true_h[-sample_size:, :]
        
        positive = tf.expand_dims(y_pred[y_true > 0], axis = -1)
        negative = tf.expand_dims(y_pred[y_true < 1], axis = -1)
        return positive, negative


    def _compute_loss(
            self,
            positive: tf.Tensor,
            negative: tf.Tensor,
            positive_h: tf.Tensor,
            negative_h: tf.Tensor
        ) -> tf.Tensor:

        loss_positive = 0.
        if tf.shape(positive)[0] > 0:
            diff = tf.reshape(negative_h, (1, -1)) - positive + self.gamma
            loss_positive += tf.math.reduce_mean(diff ** 2.)
 
        loss_negative = 0.
        if tf.shape(negative)[0] > 0:
            diff = tf.reshape(negative, (1, -1)) - positive_h + self.gamma 
            loss_negative += tf.math.reduce_mean(diff ** 2.)
    
        return loss_negative + loss_positive


    def _update_gamma(
            self
        ) -> None:

        positive, negative = self._get_posneg(use_g = True)

        diff = positive - tf.reshape(negative, (1, -1))
        bool_mask = (diff > 0)
        AUC = tf.reduce_mean(tf.cast(bool_mask, dtype = self.dtype))
        corr_ord = tf.sort(tf.reshape(diff[bool_mask], (-1,)))

        n_wrong_ord = (1 - AUC) * tf.cast(tf.size(diff), dtype = self.dtype)
        n_corr_ord = tf.cast(tf.shape(corr_ord)[0], dtype = self.dtype) - 1.
        
        idx = tf.cast(tf.math.minimum(n_wrong_ord * self.delta, n_corr_ord), 
                      dtype = tf.int32)
        self.gamma = corr_ord[idx]


##############################################################################
