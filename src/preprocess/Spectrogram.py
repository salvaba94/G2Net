# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 23:02:45 2021

@author: salva
"""

import tensorflow as tf
import numpy as np
import warnings
from scipy import signal
from typing import Tuple, Mapping, Union

from utilities import GeneralUtilities


##############################################################################


class CQTLayer(tf.keras.layers.Layer):
    """
    Constant Q Transform keras layer. Based on the nnaudio implementation 
    and extended to bound the output spectrogram to the input of an image-based
    model.
    
    See "K. W. Cheuk, H. Anderson, K. Agres and D. Herremans, nnAudio: An 
    on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D 
    Convolutional Neural Networks, in IEEE Access, vol. 8, pp. 161981-162003, 
    2020, doi: 10.1109/ACCESS.2020.3019084. 
    
    https://github.com/KinWaiCheuk/nnAudio
    """

    def __init__(
            self, 
            sample_rate: float = 2048., 
            hop_length: int = 32, 
            n_bins = 84,
            bins_per_octave: int = 12,
            f_band: Tuple[float, float] = (0., None),
            norm: int = 1,
            filter_scale: int = 1,
            window: str = "hann",
            center: bool = True, 
            pad_mode: str = "reflect",
            norm_type: str = "librosa",
            image_out: bool = True,
            perc_range: float = 0.05,
            minmax_init: Tuple[float, float] = (0, -1e7),
            tpu: bool = False,
            **kwargs
        ) -> None:
        """
        Function to initialize the object.

        Parameters
        ----------
        sample_rate : float, optional
            The sampling rate for the input time series [Hz]. It is used to 
            calculate the correct "f_min" and "f_max". The default is 2048.
        hop_length : int, optional
            The hop (or stride) size. The default is 512.
        n_bins : int, optional
            The total numbers of CQT bins. Will be ignored if "f_max" is not None. 
            The default is 32. 
        bins_per_octave : int, optional
            Number of bins per octave. The default is 12.
        f_band : Tuple[float, float], optional
            The frequency for the lowest (f_min) and highest (f_max) CQT bin [Hz]. 
            The default is (0., None). Since the default highest CQT bin frequency 
            is None, it will be inferred from n_bins and bins_per_octave. 
            If provided, n_bins will be ignored. 
        norm : int, optional
            Normalization for the CQT kernels. 1 means L1 normalization and 2 
            means L2 normalization. The default is 1, which is same as the 
            normalization used in librosa.
        filter_scale : int, optional
            Filter scale factor. Values of filter_scale smaller than 1 can be 
            used to improve the time resolution at the cost of degrading the 
            frequency resolution. The default is 1.
        window : Union[str, Tuple[str, float]], optional
            The windowing function for CQT. If it is a string, It uses 
            "scipy.signal.get_window". If it is a tuple, only the gaussian 
            window wanrantees constant Q factor. The default is "hann".
        center : bool
            Putting the CQT keneral at the center of the time-step or not.
            The default is True.
        pad_mode : str, optional
            The padding method. The default is "reflect".
            The possible options are:
                - "constant"
                - "reflect"
        norm_type : str, optional
            Type of the normalization. The default is "librosa". 
            The possible options are: 
                - "librosa" : the output fits the librosa one.
                - "convolutional" : the output conserves the convolutional 
                  inequalities of the wavelet transform.
                - "wrap" : wraps positive and negative frequencies into 
                  positive frequencies. 
        image_out : bool, optional
            Whether to return a spectrogram scaled to the 0-255 range with 
            current minimum and maximum. Default is True.
        perc_range : float, optional
            Extra range to apply to tracked spectrogram output maximum to 
            leave a safe margin for non-seen examples. The default is 0.05.
        minmax_init : Tuple[float, float], optional
            Initial values for tracking minimum and maximum spectrogram outputs.
            The default is (0., -1e7).
        tpu : str, optional
            Whether this layer is to be applied in TPU or not. The default is False. 
        """
    
        super(CQTLayer, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.bins_per_octave = bins_per_octave
        self.f_band = f_band
        self.norm = norm
        self.filter_scale = filter_scale
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.norm_type = norm_type
        self.perc_range = perc_range
        self.image_out = image_out
        self.minmax_init = minmax_init
        self.tpu = tpu

        q = np.float(filter_scale) / (2. ** (1. / bins_per_octave) - 1.)
        cqt_kernels, kernel_width, lengths, _ = _Utilities.create_cqt_kernels(
            q, sample_rate, f_band, n_bins, bins_per_octave, norm, window)

        cqt_kernels_real = np.swapaxes(cqt_kernels.real[:, np.newaxis, :], 0, -1)
        cqt_kernels_imag = np.swapaxes(cqt_kernels.imag[:, np.newaxis, :], 0, -1)
        
        self.cqt_kernels_real = tf.Variable(initial_value = cqt_kernels_real, 
                                            trainable = self.trainable,
                                            name = self.name + "/real_kernels", 
                                            dtype = self.dtype)
        self.cqt_kernels_imag = tf.Variable(initial_value = cqt_kernels_imag, 
                                            trainable = self.trainable,
                                            name = self.name + "/imag_kernels",
                                            dtype = self.dtype)

        padding = tf.constant([[0, 0], [kernel_width // 2, kernel_width // 2],
                               [0, 0]])
    
        self.padding_fn = lambda x: x
        self.padding_conv = "VALID"
        if center:
            if self.tpu:
                self.padding_conv = "SAME"
                warnings.warn("Using TPU, changing to compatible version", 
                              SyntaxWarning)
            else:
                if pad_mode == "constant":
                    self.padding_fn = lambda x: tf.pad(x, padding, mode = "CONSTANT")
                elif pad_mode == "reflect":
                    self.padding_fn = lambda x: tf.pad(x, padding, mode = "REFLECT")
                else:
                    warnings.warn("Padding method not recognised, applying no padding", 
                                  SyntaxWarning)
                
        self.norm_factor = 1.
        lengths = tf.constant(lengths, dtype = self.cqt_kernels_real.dtype)
        if norm_type == "librosa":
            self.norm_factor = tf.math.sqrt(lengths)
        elif norm_type == "convolutional":
            self.norm_factor = 1.
        elif norm_type == "wrap":
            self.norm_factor = 2.
        else:
            warnings.warn("Normalization method not recognised, \
                          applying convolutional normalization", 
                          SyntaxWarning)
                
        self.image_out = image_out
        self.max = tf.Variable(initial_value = minmax_init[-1], 
                               name = self.name + "/max", 
                               dtype = self.dtype)
        self.min = tf.Variable(initial_value = minmax_init[0], 
                               name = self.name + "/min", 
                               dtype = self.dtype)


    def build(
            self, 
            input_shape: Tuple[int, int, int]
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
            self.trainable_weights.append(self.cqt_kernels_real)
            self.trainable_weights.append(self.cqt_kernels_imag)
        else:
            self.non_trainable_weights.append(self.cqt_kernels_real)
            self.non_trainable_weights.append(self.cqt_kernels_imag)

        self.non_trainable_weights.append(self.max)
        self.non_trainable_weights.append(self.min)

        super(CQTLayer, self).build(input_shape)


    def call(
            self, 
            data: tf.Tensor,
            training: bool = None
        ) -> tf.Tensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_samples, n_detectors)
            A batch of input mono waveforms, n_detectors should be last
        training : bool, optional
            Whether the forward pass is called in training or in prediction 
            mode. Default is None.

        Returns
        -------
        tf.Tensor, shape = (None, n_time, n_freq, n_detectors)
            The corresponding batch of constant Q transforms.
        """

        CQT = []
        for i in range(data.get_shape()[-1]):
            x = data[..., i]
            x = GeneralUtilities.broadcast_dim(x)
            x = tf.cast(x, self.dtype)
            if not self.tpu:
                x = self.padding_fn(x)
            x_real = tf.nn.conv1d(x, self.cqt_kernels_real, 
                                  stride = self.hop_length, 
                                  padding = self.padding_conv)
            x_imag = -tf.nn.conv1d(x, self.cqt_kernels_imag, 
                                   stride = self.hop_length, 
                                   padding = self.padding_conv)
            x_real *= self.norm_factor
            x_imag *= self.norm_factor
            x = tf.pow(x_real, 2) + tf.pow(x_imag, 2)
            if self.trainable:
                x += 1e-8
            x = tf.math.sqrt(x)
            x = tf.transpose(x, [0, 2, 1])
            x = tf.expand_dims(x, axis = -1)
            CQT = x if (i == 0) else tf.concat([CQT, x], axis = -1)
            
        if self.image_out:
            if training:
                max_batch = tf.stop_gradient(tf.reduce_max(CQT))
                max_val = tf.stop_gradient(tf.math.maximum(self.max, max_batch))
                min_batch = tf.stop_gradient(tf.reduce_min(CQT))
                min_val = tf.stop_gradient(tf.math.minimum(self.min, min_batch))
        
                self.max.assign(max_val)
                self.min.assign(min_val)

            r_minmax = tf.stop_gradient(self.max - self.min)
            min_val = tf.stop_gradient(self.min)
            max_val = tf.stop_gradient(self.max + self.perc_range * r_minmax)
            CQT = (CQT - min_val)/(max_val - min_val)
            
            CQT = tf.clip_by_value(CQT, clip_value_min = 0., clip_value_max = 1.)

        return CQT


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
            "sample_rate": self.sample_rate,
            "n_bins": self.n_bins,
            "hop_length": self.hop_length,
            "bins_per_octave": self.bins_per_octave,
            "f_band": self.f_band,
            "norm": self.norm,
            "filter_scale": self.filter_scale,
            "window": self.window,
            "center": self.center,
            "pad_mode": self.pad_mode,
            "norm_type": self.norm_type,
            "perc_range": self.perc_range,
            "image_out": self.image_out,
            "minmax_init": self.minmax_init,
            "tpu": self.tpu
        }
        
        config.update(super(CQTLayer, self).get_config())
        return config


##############################################################################


class _Utilities(object):
    """
    Class with local auxiliary functions.
    """

    @staticmethod
    def create_cqt_kernels(
            q: float,
            sample_rate: float,
            f_band: Tuple[float, float] = (0., None),
            n_bins: int = 84,
            bins_per_octave: int = 12,
            norm: int = 1,
            window: Union[str, Tuple[float, str]] = "hann"
        ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        """
        Function to automatically create CQT kernels in time domain.
        
        Parameters
        ----------
        q : float
            Q parameter.
        sample_rate : float
            The sampling rate for the input time series. It is used to 
            calculate the correct "f_min" and "f_max". 
        f_band : Tuple[float, float], optional
            The frequency for the lowest (f_min) and highest (f_max) CQT bin [Hz]. 
            The default is (0., None). Since the default highest CQT bin frequency 
            is None, it will be inferred from n_bins and bins_per_octave. 
            If provided, n_bins will be ignored. 
        n_bins : int, optional
            The total numbers of CQT bins. Will be ignored if "f_max" is not None. 
            The default is 84. 
        bins_per_octave : int, optional
            Number of bins per octave. The default is 12.
        norm : int, optional
            Normalization for the CQT kernels. 1 means L1 normalization and 2 
            means L2 normalization. The default is 1, which is same as the 
            normalization used in librosa.
        window : Union[str, Tuple[float, str]], optional
            The windowing function for CQT. If it is a string, It uses 
            "scipy.signal.get_window". If it is a tuple, only the gaussian 
            window wanrantees constant Q factor. The default is "hann".

        Raises
        ------
        ValueError
            If maximum bins frequency is greater than the Nyquist frequency.

        Returns
        -------
        Tuple[np.ndarray, int, np.ndarray, np.ndarray]
            CQT kernels, length of the frequency bins and associated 
            frequencies.
        """

        f_min, f_max = f_band[0], f_band[-1]

        len_min = np.ceil(q * sample_rate / f_min)
        fft_len = 2 ** np.int(np.ceil(np.log2(len_min)))
    
        if (f_max is not None) and (n_bins is None):
            n_bins = np.ceil(bins_per_octave * np.log2(f_max / f_min))
            freqs = f_min * 2. ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        elif (f_max is None) and (n_bins is not None):
            freqs = f_min * 2. ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        else:
            warnings.warn("If f_max is given, n_bins will be ignored", SyntaxWarning)
            n_bins = np.ceil(bins_per_octave * np.log2(f_max / f_min))
            freqs = f_min * 2. ** (np.r_[0:n_bins] / np.float(bins_per_octave))

        f_nyq = sample_rate / 2.
        if np.max(freqs) > f_nyq:
            raise ValueError(f"The top bin {np.max(freqs)} Hz has exceeded \
                             the Nyquist frequency, please reduce `n_bins`")

        kernel = np.zeros((np.int(n_bins), np.int(fft_len)), dtype = np.complex64)
    
        lengths = np.ceil(q * sample_rate / freqs)
        for k in range(np.int(n_bins)):
            freq = freqs[k]
            l = np.ceil(q * sample_rate / freq)

            if l % 2 == 1:
                start = np.int(np.ceil(fft_len / 2. - l / 2.)) - 1
            else:
                start = np.int(np.ceil(fft_len / 2. - l / 2.))
    
            sig = signal.get_window(window, np.int(l), fftbins = True)
            sig = sig * np.exp(np.r_[-l // 2:l // 2] * 1j * 2 * np.pi * \
                               freq / sample_rate) / l
            
            if norm:
                kernel[k, start:start + np.int(l)] = sig / np.linalg.norm(sig, norm)
            else:
                kernel[k, start:start + np.int(l)] = sig

        return kernel, fft_len, lengths, freqs

    
##############################################################################


