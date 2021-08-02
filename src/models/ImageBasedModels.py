# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:27:02 2021

@author: salva
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB1
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import EfficientNetB5
from tensorflow.keras.applications.efficientnet import EfficientNetB6
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from typing import Tuple
from functools import partial

from automl.efficientnetv2 import effnetv2_model

##############################################################################

class G2NetEfficientNet(object):
    """
    G2Net model for images based on EfficientNet
    """
    def __init__(
            self, 
            input_shape: Tuple[int, int, int],
            weights: str = "imagenet",
            dtype: type = tf.float32
        ) -> None:


        self.effnet_models = {
            "B0v1": EfficientNetB0,
            "B1v1": EfficientNetB1,
            "B2v1": EfficientNetB2,
            "B3v1": EfficientNetB3,
            "B4v1": EfficientNetB4,
            "B5v1": EfficientNetB5,
            "B6v1": EfficientNetB6,
            "B7v1": EfficientNetB7,
            "B8v1": partial(effnetv2_model.get_model, model_name = "efficientnet-b8"), 
            "L2v1": partial(effnetv2_model.get_model, model_name = "efficientnet-l2"), 
            "Sv2":  partial(effnetv2_model.get_model, model_name = "efficientnetv2-s"), 
            "Mv2":  partial(effnetv2_model.get_model, model_name = "efficientnetv2-m"), 
            "Lv2":  partial(effnetv2_model.get_model, model_name = "efficientnetv2-l"),  
            "XLv2": partial(effnetv2_model.get_model, model_name = "efficientnetv2-xl"), 
            "B0v2": partial(effnetv2_model.get_model, model_name = "efficientnetv2-b0"), 
            "B1v2": partial(effnetv2_model.get_model, model_name = "efficientnetv2-b1"), 
            "B2v2": partial(effnetv2_model.get_model, model_name = "efficientnetv2-b2"), 
            "B3v2": partial(effnetv2_model.get_model, model_name = "efficientnetv2-b3") 
        }
        self.weights = weights
        self.input_shape = input_shape

        self.input = Input(shape = input_shape, dtype = dtype, name = "input")
        self.flatten = Flatten()
        self.dense = Dense(units = 1, activation = "sigmoid")


    def get_model(
            self,
            effnet_id: str = "B0v1"
        ) -> tf.keras.Model:

        x = self.input
        y = self.effnet_models[effnet_id](include_top = False, 
                 weights = self.weights, input_shape = self.input_shape)(x)
        y = self.flatten(y)
        y = self.dense(y)
        return tf.keras.Model(inputs = [x], outputs = [y])


# class LogMelEfficientNetB2(tf.keras.Model):
#     def __init__(self, n_samples, n_detect, sample_rate, fft_size, hop_size, 
#                  n_mels, **kwargs):
#         super(LogMelEfficientNetB2, self).__init__(**kwargs)
#         self.logmelspec = LogMelSpectrogramLayer(sample_rate, fft_size, hop_size, n_mels)
#         self.effnetb2 = EfficientNetB2(include_top = False, weights = None, 
#                         input_shape = (n_samples // hop_size, n_mels, n_detect))
#         self.flatten = Flatten()
#         self.dense = Dense(units = 1, activation = "sigmoid")
#
#     def call(self, inputs):
#         out = self.logmelspec(inputs)
#         out = self.effnetb2(out)
#         out = self.flatten(out)
#         out = self.dense(out)
#         return out


# def LogMelEfficientNetB2(n_samples, n_detect, sample_rate, fft_size, 
#                           hop_size, n_mels):
#     x = Input(shape = (n_samples, n_detect), dtype = tf.float64)
#     y = LogMelSpectrogram(sample_rate, fft_size, hop_size, n_mels)(x)
#     y = EfficientNetB2(include_top = False, weights = None, 
#                         input_shape = (n_samples // hop_size, n_mels, n_detect)
#                         )(y)
#     y = Flatten()(y)
#     y = Dense(units = 1, activation = "sigmoid")(y)
#     return Model(inputs = [x], outputs = [y])