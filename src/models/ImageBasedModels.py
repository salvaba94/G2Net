# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:27:02 2021

@author: salva
"""

import tensorflow as tf
import copy
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from typing import Tuple
from preprocess import CQTLayer, TukeyWinLayer, BandpassLayer, WindowingLayer
from preprocess import PermuteChannel, SpectralMask, TimeMask, FreqMask

from automl.efficientnetv2 import effnetv2_model
from automl.efficientnetv2 import effnetv2_configs
from automl.efficientnetv2 import hparams


##############################################################################

class G2NetEfficientNet(object):
    """
    G2Net model for images based on EfficientNet
    """
    def __init__(
            self, 
            input_shape: Tuple[int, int],
            window_shape: float = 0.2,
            trainable_window: bool = False,
            sample_rate: float = 2048., 
            degree_filt: int = 8,
            f_band_filt: Tuple[float, float] = (20., 500.),
            trainable_filt: bool = False,
            hop_length: int = 64,
            f_band_spec: Tuple[float, float] = (20., 500.),
            bins_per_octave: int = 12,
            window_cqt: str = "hann",
            perc_range: float = 0.05, 
            trainable_cqt: bool = False,
            resize_shape: Tuple[int, int] = (128, 128),
            p_perm: float = 0.1,
            p_mask: float = 0.1,
            n_max_mask_t: int = 2,
            w_mask_t: Tuple[int, int] = (12, 25),
            n_max_mask_f: int = 2,
            w_mask_f: Tuple[int, int] = (12, 25),
            dtype: type = tf.float32,
            strategy: str = "GPU"
        ) -> None:
        """
        Function to initialise the object.
        
        Parameters
        ----------
        input_shape : Tuple[int, int], 
            Shape of the input to the model without accounting for batch size.
        window_shape : float, optional
            Shape parameter of the Tukey temporal window. The default is 0.2.
        trainable_window : bool, optional
            Whether the Tukey temporal window should be trained or not. 
            The default is False.
        sample_rate : float, optional
            The sampling rate for the input time series. The default is 2048.
        degree_filt : int, optional
            Degree of the bandpass filter. The default is 8.
        f_band_filt : Tuple[float, float], optional
            The frequency band for the bandpass filter [Hz]. The default 
            is (20, 500).
        trainable_filt : bool, optional
            Whether the bandpass filter should be trained or not. 
            The default is False.
        hop_length : int, optional
            The hop (or stride) size for the CQT layer. The default is 512.
        f_band_spec : Tuple[float, float], optional
            The frequency for the lowest (f_min) and highest (f_max) CQT bins [Hz]. 
            The default is (20, 500).
        bins_per_octave : int, optional
            Number of bins per octave for the CQT layer. The default is 12.
        window_cqt : str, optional
            The windowing function for CQT. The default is "hann".
        perc_range : float, optional
            Extra range to apply to tracked spectrogram output maximum to 
            leave a safe margin for non-seen examples. The default is 0.05.
        trainable_cqt : bool, optional
            Whether the cqt layer should be trained or not. If transfer learning 
            is applied, the recommendation is to freeze this layer during the 
            first epochs and activate its training afterwards. The default is False.
        resize_shape : Tuple[int, int], optional
            Spectrogram resize shape without including batch size and channels. 
            The default is (128, 128).
        p_perm : float, optional
            Probability of performing a channel permutation for regularisation.
            The default is 0.1.
        p_mask : foat, optional
            Probability of performing spectral mask for regularisation. The 
            default is 0.1.
        n_max_mask_t : int, optional
            Maximum number of masks in time dimension. The default is 2.
        w_mask_t : Tuple[int, int], optional
            Minimum and maximum width of masking bands in the time dimension. 
            The default is (12, 25).
        n_max_mask_f : int, optional
            Maximum number of masks in frequency dimension. The default is 2.
        w_mask_f : Tuple[int, int], optional
            Minimum and maximum width of masking bands in the frequency dimension. 
            The default is (12, 25).
        dtype : type, optional
            Data type of the model layer parameters. The default is tf.float32.
        strategy : str, optional
            In use strategy. It is mainly used to switch to layers compatible 
            with XLA when using TPU. The default is "GPU".
            Available options are:
                - "TPU"
                - "GPU"
                - "CPU"
        """

        self.input_shape = input_shape
        self.resize_shape = resize_shape
        self.strategy = strategy
        self.n_max_mask_t = n_max_mask_t
        self.n_max_mask_f = n_max_mask_f

        self.input = Input(shape = input_shape, dtype = dtype, name = "input")

        if self.strategy == "TPU":
            self.window = WindowingLayer(window = ("tukey", window_shape),
                                         window_len = input_shape[0],
                                         trainable = trainable_window, name = "window")
        else:
            self.window = TukeyWinLayer(initial_alpha = window_shape, 
                                        trainable = trainable_window, name = "window")

        self.bandpass = BandpassLayer(sample_rate = sample_rate, degree = degree_filt, 
                                      f_band = f_band_filt, n_samples = input_shape[0], 
                                      trainable = trainable_filt, name = "bandpass")
        
        tpu = True if strategy == "TPU" else False
        self.cqt = CQTLayer(sample_rate = sample_rate, hop_length = hop_length, 
                            f_band = f_band_spec, bins_per_octave = bins_per_octave,
                            window = window_cqt, trainable = trainable_cqt, 
                            perc_range = perc_range, tpu = tpu, name = "cqt")

        self.resize = Resizing(resize_shape[0], resize_shape[1], name = "resize")

        self.permute = PermuteChannel(p = p_perm, name = "permute")
        
        if self.strategy == "TPU":
            self.mask_t = TimeMask(p = p_mask, w_mask = w_mask_t, name = "mask_t")
            self.mask_f = FreqMask(p = p_mask, w_mask = w_mask_f, name = "mask_f")
        else:
            self.mask = SpectralMask(p = p_mask, n_max_mask_t = self.n_max_mask_t,
                                     w_mask_t = w_mask_t, n_max_mask_f = self.n_max_mask_f,
                                     w_mask_f = w_mask_f, name = "mask")

        self.flatten = Flatten(name = "flatten")
        self.dense = Dense(units = 1, activation = "sigmoid", name = "dense")


    def get_model(
            self,
            effnet_id: str = "efficientnetv2-b2",
            weights: str = "imagenet",
        ) -> tf.keras.Model:
        """
        Function to get the model object.

        Parameters
        ----------
        effnet_id : str, optional
            Id of the efficientnet backend model to use. The default is "efficientnetv2-b2".
            Available options are:
                - "efficientnetv2-s"
                - "efficientnetv2-m"
                - "efficientnetv2-l"
                - "efficientnetv2-xl"
                - "efficientnetv2-b0"
                - "efficientnetv2-b1"
                - "efficientnetv2-b2"
                - "efficientnetv2-b3"
                - "efficientnet-b0"
                - "efficientnet-b1"
                - "efficientnet-b2"
                - "efficientnet-b3"
                - "efficientnet-b4"
                - "efficientnet-b5"
                - "efficientnet-b6"
                - "efficientnet-b7"
                - "efficientnet-b8"
                - "efficientnet-l2"
        weights : str, optional
            Whether to use weights from pre-trained models or not. The default 
            is "imagenet". Available options are:
                - "imagenet"
                - "imagenet21k"
                - "imagenet21k-ft1k"
                - "jft"
        """

        effnet_config = copy.deepcopy(hparams.base_config)
        effnet_config.override(effnetv2_configs.get_model_config(effnet_id))
        if self.strategy == "TPU" and not effnet_config.model.bn_type:
            effnet_config.model.bn_type = "tpu_bn"

        effnet = effnetv2_model.get_model(model_name = effnet_id,
            model_config = effnet_config.model, include_top = False, 
            weights = weights)

        x = self.input

        y = self.window(x)
        y = self.bandpass(y)
        y = self.cqt(y)

        y = self.resize(y)

        y = self.permute(y)

        if self.strategy == "TPU":
            for _ in range(self.n_max_mask_t):
                y = self.mask_t(y)
            for _ in range(self.n_max_mask_f):
                y = self.mask_f(y)
        else:
            y = self.mask(y)

        y = effnet(y)
        y = self.flatten(y)
        y = self.dense(y)
        return tf.keras.Model(inputs = [x], outputs = [y])


##############################################################################
