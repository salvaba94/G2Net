# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:20:47 2021

@author: salva
"""

import tensorflow as tf
from typing import Mapping
from .GeneralUtilities import GeneralUtilities


##############################################################################

class LogMelSpectrogramLayer(tf.keras.layers.Layer):
    """
    Log-magnitude mel-scaled spectrogram keras layer.
    """

    def __init__(
            self, 
            sample_rate, 
            fft_size, 
            hop_size, 
            n_mels,
            f_min = 0.0, 
            f_max = None, 
            **kwargs
        ) -> None:
        """
        Function to initialise the object.

        Parameters
        ----------
        sample_rate : TYPE
            DESCRIPTION.
        fft_size : TYPE
            DESCRIPTION.
        hop_size : TYPE
            DESCRIPTION.
        n_mels : TYPE
            DESCRIPTION.
        f_min : TYPE, optional
            DESCRIPTION. The default is 0.0.
        f_max : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None
        """
        super(LogMelSpectrogramLayer, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins = self.n_mels,
            num_spectrogram_bins = fft_size // 2 + 1,
            sample_rate = self.sample_rate,
            lower_edge_hertz = self.f_min,
            upper_edge_hertz = self.f_max)

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super(LogMelSpectrogramLayer, self).build(input_shape)


    def call(
            self, 
            waveforms: tf.Tensor
        ) -> tf.Tensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        waveforms : tf.Tensor, shape = (None, n_samples, n_detectors)
            A batch of input mono waveforms.

        Returns
        -------
        tf.Tensor, shape = (None, n_time, n_freq, n_detectors)
            The corresponding batch of log-scaled Mel-spectrograms.
        """

        waveforms_t = tf.transpose(waveforms, perm = [0, 2, 1])
        waveforms_sc = waveforms_t / tf.reduce_max(waveforms, axis = -1, 
                                                   keepdims=True)
        spec = tf.signal.stft(waveforms_sc,
                              frame_length = self.fft_size,
                              frame_step = self.hop_size, 
                              pad_end = True)
        magnitude_spec = tf.abs(spec)
        mel_spec = tf.matmul(magnitude_spec ** 2., self.mel_filterbank)
        log_mel_spec = LogMelSpectrogramLayer.__power_to_db(mel_spec)
        scal_log_mel_spec = GeneralUtilities.scale_linearly(log_mel_spec)
        image_log_mel_spec = tf.transpose(scal_log_mel_spec, perm = [0, 2, 3, 1])

        return image_log_mel_spec

    @staticmethod
    def __tf_log10(
            x: tf.Tensor
        ) -> tf.Tensor:
        """
        This function calculates the log10 using parallelisable functions.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor to compute the log10.

        Returns
        -------
        tf.Tensor
            Log10 of the input tensor.

        """
        return tf.math.log(x) / tf.math.log(10.)


    @staticmethod
    def __power_to_db(magnitude, amin = 1e-16, top_db = 80.):
        """
        Function to transform from signal power to db
        """
        ref_value = tf.reduce_max(magnitude)
        log_spec = 10. * (
            LogMelSpectrogramLayer.__tf_log10(tf.maximum(amin, magnitude)) - 
            LogMelSpectrogramLayer.__tf_log10(tf.maximum(amin, ref_value)))
        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
        return log_spec


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
            "fft_size": self.fft_size,
            "hop_size": self.hop_size,
            "n_mels": self.n_mels,
            "sample_rate": self.sample_rate,
            "f_min": self.f_min,
            "f_max": self.f_max,
        }
        
        config.update(super(LogMelSpectrogramLayer, self).get_config())
        return config
    