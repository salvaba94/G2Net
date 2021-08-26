# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:27:02 2021

@author: salva
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda
from tensorflow.keras.layers.experimental.preprocessing import Resizing
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
from preprocess import CQTLayer, TukeyWinLayer, BandpassLayer

from automl.efficientnetv2 import effnetv2_model


##############################################################################

class G2NetEfficientNet(object):
    """
    G2Net model for images based on EfficientNet
    """
    def __init__(
            self, 
            input_shape: Tuple[int, int],
            window_shape: float = 0.25,
            trainable_window: bool = False,
            sample_rate: float = 2048., 
            degree_filt: int = 8,
            f_band_filt: Tuple[float, float] = (20., 1024.),
            trainable_filt: bool = False,
            hop_length: int = 64,
            f_band_spec: Tuple[float, float] = (20., 1024.),
            bins_per_octave: int = 12,
            window_cqt: str = "hann",
            trainable_cqt: bool = False,
            resize_shape: Tuple[int, int] = (128, 128),
            weights: str = "imagenet",
            perc_range: float = 0.01, 
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

        self.input_shape = input_shape
        self.resize_shape = resize_shape
        self.weights = weights
        self.input = Input(shape = input_shape, dtype = dtype, name = "input")
        self.window = TukeyWinLayer(initial_alpha = window_shape, 
                                    trainable = trainable_window, name = "windowing")
        self.filter = BandpassLayer(degree = degree_filt, sample_rate = sample_rate, 
                                    f_band = f_band_filt, trainable = trainable_filt, 
                                    n_samples = input_shape[0], name = "bandpassing")
        self.cqt = CQTLayer(sample_rate = sample_rate, hop_length = hop_length, 
                            f_band = f_band_spec, bins_per_octave = bins_per_octave,
                            window = window_cqt, trainable = trainable_cqt, 
                            perc_range = perc_range, name = "cqt")
        self.clip = Lambda(lambda x: tf.clip_by_value(x, clip_value_min = 0.,
                           clip_value_max = 255.), name = "clipping")
        self.resize = Resizing(resize_shape[0], resize_shape[1], name = "resizing")

        self.flatten = Flatten(name = "flatten")
        self.dense = Dense(units = 1, activation = "sigmoid", name = "dense")


    def get_model(
            self,
            effnet_id: str = "B0v1"
        ) -> tf.keras.Model:

        effnet = self.effnet_models[effnet_id]

        x = self.input
        y = self.window(x)
        y = self.filter(y)
        y = self.cqt(y)
        y = self.clip(y)
        y = self.resize(y)
        y = effnet(include_top = False, weights = self.weights, 
                   input_shape = self.resize_shape + (self.input_shape[-1],))(y)
        y = self.flatten(y)
        y = self.dense(y)
        return tf.keras.Model(inputs = [x], outputs = [y])


##############################################################################
