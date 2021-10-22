# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:44:02 2021

@author: salva
"""

import tensorflow as tf
import numpy as np
from scipy import signal
from scipy import interpolate
from functools import partial
from typing import Tuple, Union, Mapping

from utilities import GeneralUtilities


##############################################################################


class TukeyWinLayer(tf.keras.layers.Layer):
    """
    Layer that applies a Tukey window function to an input time series, where 
    the possibility of training the shape parameter is given. Not usable with TPU.
    """

    def __init__(
            self, 
            initial_alpha: float = 0.25,
            **kwargs
        ) -> None:
        """
        Function to initialise the object.

        Parameters
        ----------
        initial_alpha : float, optional
            Shape parameter of the tukey window. The default is 0.25.
        """
    
        super(TukeyWinLayer, self).__init__(**kwargs)

        self.alpha = tf.Variable(initial_value = initial_alpha, 
                                 trainable = self.trainable,
                                 name = self.name + "/alpha", 
                                 dtype = self.dtype)


    def build(
            self, 
            input_shape: Tuple[int, int]
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
            self.trainable_weights.append(self.alpha)
        else:
            self.non_trainable_weights.append(self.alpha)
        super(TukeyWinLayer, self).build(input_shape)


    def call(
            self, 
            data: tf.Tensor
        ) -> tf.Tensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_samples, n_detectors)
            A batch of input waveforms, n_detectors (n_channels) should 
            be last.

        Returns
        -------
        tf.Tensor, shape = (None, n_samples, n_detectors)
            The corresponding batch of windowed signals.
        """

        x = GeneralUtilities.broadcast_dim(data)
        x = tf.cast(x, self.dtype)
        w_len = tf.shape(x)[1]
        window = GeneralUtilities.broadcast_dim(self._get_window(w_len))
        x *= window
        return x


    def _get_ones(
            self,
            w_len: int
        ) -> tf.Tensor:
        """
        Case for a null shape parameter.

        Parameters
        ----------
        w_len : int
            Length of the window.

        Returns
        -------
        tf.Tensor : shape = (window_len,)
            A window of all ones.
        """
 
        window = tf.ones(w_len, dtype = self.dtype)
        return window

        
    def _get_hann(
            self,
            w_len: int
        ) -> tf.Tensor:
        """
        Case for a unity shape parameter.

        Parameters
        ----------
        w_len : int
            Length of the window.

        Returns
        -------
        tf.Tensor : shape = (window_len,)
            A hann window.
        """
 
        window = tf.signal.hann_window(w_len, periodic = False)
        window = tf.cast(window, dtype = self.dtype)
        return window


    def _get_tukey(
            self,
            w_len: int
        ) -> tf.Tensor:
        """
        Case for a non-null and non-unity shape parameter.

        Parameters
        ----------
        w_len : int
            Length of the window.

        Returns
        -------
        tf.Tensor : shape = (window_len,)
            A tukey window.
        """
 
        w_len_f = tf.cast(w_len, self.dtype)
       
        n = tf.range(0, w_len)
        width = tf.math.floor(self.alpha * (w_len_f - 1.)/2.)
        width = tf.minimum(width, w_len_f)
        width = tf.maximum(width, 0.)
        width = tf.cast(width, dtype = tf.int32)

        n_1 = tf.cast(n[:width + 1], dtype = self.dtype)
        n_2 = tf.cast(n[width + 1: w_len - width - 1], dtype = self.dtype)
        n_3 = tf.cast(n[w_len - width - 1:], dtype = self.dtype)
    
        window_1 = 0.5 * (1. + tf.math.cos(np.pi * (
            -1. + 2. * n_1 / self.alpha / (w_len_f - 1.))))
        window_2 = tf.ones(tf.shape(n_2))
        window_3 = 0.5 * (1. + tf.math.cos(np.pi * (
            - 2./self.alpha + 1. + 2. * n_3 / self.alpha / (w_len_f - 1.))))
    
        window = tf.concat((window_1, window_2, window_3), axis = 0)
        return window
    
    
    def _get_window(
            self,
            w_len: int
        ) -> tf.Tensor:
        """
        Tukey window getter.

        Parameters
        ----------
        w_len : int
            Length of the window.

        Returns
        -------
        tf.Tensor : shape = (window_len,)
            A tukey window handling the cases of null or unity shape parameters.
        """
        get_ones = partial(self._get_ones, w_len = w_len + 1)
        get_hann = partial(self._get_hann, w_len = w_len + 1)
        get_tukey = partial(self._get_tukey, w_len = w_len + 1)

        window = tf.case([(tf.less_equal(self.alpha, 0.), get_ones),
                          (tf.greater_equal(self.alpha, 1.), get_hann)], 
                         default = get_tukey)
        window = window[:-1]
        return window
    

##############################################################################


class WindowingLayer(tf.keras.layers.Layer):
    """
    Layer that applies a window function to an input time series.
    """

    def __init__(
            self, 
            window: Union[str, Tuple[str, float]] = ("tukey", 0.1),
            window_len: int = 4096,
            **kwargs
        ) -> None:
        """
        Function to initialise the object.

        Parameters
        ----------
        window : str or Tuple[str, float], optional
            The type of window to create with any parameter it might need. 
            The default is "tukey" with alpha 0.1.
        window_len : int, optional
            The number of samples in the window (set it to the signal length). 
            The default is 4096.
        """
    
        super(WindowingLayer, self).__init__(**kwargs)

        self.window = window
        self.window_len = window_len

        sig = signal.get_window(window, window_len)[np.newaxis, :, np.newaxis]
        self.window = tf.Variable(initial_value = sig, trainable = self.trainable,
                                  name = self.name + "/window", dtype = self.dtype)


    def build(
            self, 
            input_shape: Tuple[int, int]
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
            self.trainable_weights.append(self.window)
        else:
            self.non_trainable_weights.append(self.window)
        super(WindowingLayer, self).build(input_shape)


    def call(
            self, 
            data: tf.Tensor
        ) -> tf.Tensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_samples, n_detectors)
            A batch of input mono waveforms, n_detectors (n_channels) should 
            be last.

        Returns
        -------
        tf.Tensor, shape = (None, n_samples, n_detectors)
            The corresponding batch of windowed waveforms.
        """

        x = GeneralUtilities.broadcast_dim(data)
        x = tf.cast(x, self.dtype)
        x *= self.window
        return x


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
            "window" : self.window,
            "window_len" : self.window_len
        }
        
        config.update(super(WindowingLayer, self).get_config())
        return config


##############################################################################


class WhitenLayer(tf.keras.layers.Layer):
    """
    Layer that applies spectral whitening to an input time series.
    """

    def __init__(
            self,
            sample_rate: float = 2048.,
            **kwargs
        ) -> None:
        """
        Function to initialise the object.
        
        Parameters
        ----------
        sample_rate : float, optional
            Sample rate of the signal [Hz]. Default value is 2048.
        """
        super(WhitenLayer, self).__init__(**kwargs)
        self.norm = np.sqrt(sample_rate)


    def build(
            self, 
            input_shape: Tuple[int, int]
        ) -> None:
        """
        Function to build the graph of the layer. Adds trainable and non-
        trainable parameters.

        Parameters
        ----------
        input_shape : Tuple[int, int]
            Shape of the input to the layer.
        """

        super(WhitenLayer, self).build(input_shape)


    def call(
            self, 
            data: tf.Tensor
        ) -> tf.Tensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_samples, n_detectors)
            A batch of input mono waveforms, n_detectors (n_channels) should 
            be last.

        Returns
        -------
        tf.Tensor, shape = (None, n_samples, n_detectors)
            The corresponding whitened signals.
        """
        data_ref = GeneralUtilities.broadcast_dim(data)

        whiten = []
        for i in range(data.get_shape()[-1]):
            x = data_ref[..., i]
            x = tf.cast(x, tf.complex64)
    
            spec = tf.signal.fft(x)

            real = tf.cast(tf.math.real(spec), tf.complex64)
            imag = tf.cast(tf.math.imag(spec), tf.complex64) 
            spec_conj = real - imag * 1j
            psd = spec * spec_conj
            psd = tf.cast(tf.math.real(psd), tf.complex64)
    
            x = spec / tf.math.sqrt(psd)
            x = tf.signal.ifft(x)
            x = tf.cast(tf.math.real(x) * self.norm, self.dtype)
            x = tf.expand_dims(x, axis = -1)
            whiten = x if (i == 0) else tf.concat([whiten, x], axis = -1)

        return whiten


##############################################################################


class BandpassLayer(tf.keras.layers.Layer):
    """
    Layer that applies a bandpass Butterworth filter in the frequency domain, 
    where the possibility of training frequency response from the filter is 
    given.
    """

    def __init__(
            self, 
            sample_rate: float = 2048.,
            degree: int = 8,
            f_band: Tuple[float, float] = (20., 500.),
            n_samples: int = 4096,
            **kwargs
        ) -> None:
        """
        Funtion to initialize the object.

        Parameters
        ----------
        sample_rate : float, optional
            The sampling rate for the input time series [Hz]. It is used to 
            calculate the correct "f_min" and "f_max". The default is 2048.
        degree : int, optional
            Degree of the Butterworth filter. The default is 8.
        f_band : Tuple[float, float], optional
            The frequency band for the bandpass filter [Hz]. 
            The default is (20., 500). 
        n_samples : int, optional
            Number of samples of the signal to filter. The default is 4096.
        """
    
        super(BandpassLayer, self).__init__(**kwargs)
        
        self.sample_rate = sample_rate
        self.degree = degree
        self.f_band = f_band
        self.n_samples = n_samples

        if f_band[-1] <= f_band[0]:
            raise ValueError("Maximum frequency in spectral band should be \
                             higher than minimum frequency")

        f_nyq = sample_rate / 2.
        f_min = f_band[0] / f_nyq
        f_max = f_band[-1] / f_nyq
        self.norm = tf.constant(np.sqrt((f_band[-1] - f_band[0]) / f_nyq), 
                                dtype = self.dtype)

        f_fft = np.fft.rfftfreq(n_samples, d = 1./sample_rate)

        b, a = signal.butter(degree, (f_min, f_max), btype = "bandpass")
        w, gain = signal.freqz(b, a, worN = 2 * f_fft.shape[0])
        f = (f_nyq / np.pi) * w
        gain = np.abs(gain)

        gain_f = interpolate.interp1d(f, gain, fill_value = "extrapolate")
        f_response = gain_f(f_fft)
        f_response = f_response[np.newaxis, :]

        self.f_response = tf.Variable(initial_value = f_response, 
                                 trainable = self.trainable,
                                 name = self.name + "/f_response", 
                                 dtype = self.dtype)


    def build(
            self, 
            input_shape: Tuple[int, int]
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
            self.trainable_weights.append(self.f_response)
        else:
            self.non_trainable_weights.append(self.f_response)
        super(BandpassLayer, self).build(input_shape)


    def call(
            self, 
            data: tf.Tensor
        ) -> tf.Tensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_samples, n_detectors)
            A batch of input waveforms, n_detectors (n_channels) should 
            be last.

        Returns
        -------
        tf.Tensor, shape = (None, n_samples, n_detectors)
            The corresponding batch of bandpassed signals.
        """

        data_ref = GeneralUtilities.broadcast_dim(data)
        bandpass = []
        for i in range(data_ref.get_shape()[-1]):
            x = data_ref[..., i]
            x = tf.cast(x, self.dtype)
            spec = tf.signal.rfft(x)
            spec *= tf.cast(self.f_response, tf.complex64)
            x = tf.signal.irfft(spec)
            x = tf.cast(x, self.dtype)
            x = tf.expand_dims(x, axis = -1)
            bandpass = x if (i == 0) else tf.concat([bandpass, x], axis = -1)
        
        bandpass /= self.norm
        return bandpass


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
            "degree": self.degree,
            "f_band": self.f_band,
            "n_samples": self.n_samples
        }

        config.update(super(BandpassLayer, self).get_config())
        return config

    
##############################################################################