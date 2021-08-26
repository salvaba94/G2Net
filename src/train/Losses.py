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
        
        tf.config.experimental_run_functions_eagerly(True)

        super(RocLoss, self).__init__(**kwargs)        
        self.gamma = gamma
        self.p = p


    def call(
            self, 
            y_true: tf.Tensor, 
            y_pred: tf.Tensor,
        ) -> tf.Tensor:
        
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
    Wilcoxon-Mann-Whitney Statistic
    
    See "Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the 
    Wilcoxon-Mann-Whitney Statistic."
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


# class RocStarLossTorch(torch.nn.Module):
#     """Smooth approximation for ROC AUC
#     """
#     def __init__(self, delta = 1.0, sample_size = 10, sample_size_gamma = 10, 
#                  update_gamma_each=10):
#         """
#         Args:
#             delta: Param from article
#             sample_size (int): Number of examples to take for ROC AUC approximation
#             sample_size_gamma (int): Number of examples to take for Gamma parameter approximation
#             update_gamma_each (int): Number of steps after which to recompute gamma value.
#         """
#         super().__init__()
#         self.delta = delta
#         self.sample_size = sample_size
#         self.sample_size_gamma = sample_size_gamma
#         self.update_gamma_each = update_gamma_each
#         self.steps = 0
#         size = max(sample_size, sample_size_gamma)

#         # Randomly init labels
#         np.random.seed(20)
#         self.y_pred_history = torch.from_numpy(np.random.random((size, 1)))
#         self.y_true_history = torch.from_numpy(np.random.randint(0, 2, (size, 1)))
        

#     def forward(self, y_pred, y_true):
#         """
#         Args:
#             y_pred: Tensor of model predictions in [0, 1] range. Shape (B x 1)
#             y_true: Tensor of true labels in {0, 1}. Shape (B x 1)
#         """
#         #y_pred = _y_pred.clone().detach()
#         #y_true = _y_true.clone().detach()
#         if self.steps % self.update_gamma_each == 0:
#             self.update_gamma()
#         self.steps += 1
        
#         positive = y_pred[y_true > 0]
#         negative = y_pred[y_true < 1]

#         # Take last `sample_size` elements from history
#         y_pred_history = self.y_pred_history[- self.sample_size:]
#         y_true_history = self.y_true_history[- self.sample_size:]
        
        
#         positive_history = y_pred_history[y_true_history > 0]
#         negative_history = y_pred_history[y_true_history < 1]
        
#         print("positive, negative torch: ", positive.size(0), negative.size(0))
        
#         if positive.size(0) > 0:
#             diff = negative_history.view(1, -1) + self.gamma - positive.view(-1, 1)
#             loss_positive = (torch.nn.functional.relu(diff) ** 2.).mean()
#         else:
#             loss_positive = 0
 
#         if negative.size(0) > 0:
#             diff = negative.view(1, -1) + self.gamma - positive_history.view(-1, 1)
#             loss_negative = (torch.nn.functional.relu(diff) ** 2.).mean()
#         else:
#             loss_negative = 0
            
#         loss = loss_negative + loss_positive
        
#         # Update FIFO queue
#         batch_size = y_pred.size(0)
#         print(self.y_pred_history[batch_size:].size())
#         print(y_pred.size())
#         self.y_pred_history = torch.cat((self.y_pred_history[batch_size:], y_pred.clone().detach()))
#         self.y_true_history = torch.cat((self.y_true_history[batch_size:], y_true.clone().detach()))
#         print(self.y_pred_history.size())
#         return loss

#     def update_gamma(self):
#         # Take last `sample_size_gamma` elements from history
#         y_pred = self.y_pred_history[- self.sample_size_gamma:]
#         y_true = self.y_true_history[- self.sample_size_gamma:]
        
#         positive = y_pred[y_true > 0]
#         negative = y_pred[y_true < 1]
        
#         # Create matrix of size sample_size_gamma x sample_size_gamma
#         diff = positive.view(-1, 1) - negative.view(1, -1)
#         AUC = (diff > 0).type(torch.float).mean()
#         num_wrong_ordered = (1 - AUC) * diff.flatten().size(0)
        
    
#         # Adjuct gamma, so that among correct ordered samples `delta * num_wrong_ordered` were considered
#         # ordered incorrectly with gamma added
#         correct_ordered = diff[diff > 0].flatten().sort().values
#         idx = min(int(num_wrong_ordered * self.delta), len(correct_ordered)-1)
#         self.gamma = correct_ordered[idx]
