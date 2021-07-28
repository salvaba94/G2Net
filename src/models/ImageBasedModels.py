# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:27:02 2021

@author: salva
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications.efficientnet import EfficientNetB2


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