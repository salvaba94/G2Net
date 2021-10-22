# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 10:02:11 2021

@author: salva
"""

import tensorflow as tf
from functools import partial
from typing import Tuple, Mapping



##############################################################################


class PermuteChannel(tf.keras.layers.Layer):
    """
    Layer that randomly permutes the channels from data to avoid overfitting 
    in the context of G2Net.
    """

    def __init__(
            self, 
            p: float = 0.1,
            **kwargs
        ) -> None:
        """
        Function to initialise the object.

        Parameters
        ----------
        rate : float, optional
            Probability of performing a permutation. The default 0.5.
        """
    
        super(PermuteChannel, self).__init__(**kwargs)
        self.p = p


    def build(
            self, 
            input_shape: Tuple[int, int]
        ) -> None:
        """
        Function to build the graph of the layer. Adds trainable and non-
        trainable parameters if any.

        Parameters
        ----------
        input_shape : Tuple[int, int]
            Shape of the input to the layer.
        """

        super(PermuteChannel, self).build(input_shape)



    def call(
            self, 
            data: tf.Tensor,
            training: bool = None
        ) -> tf.Tensor:
        """
        Forward pass of the layer (requires channels to be last).

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_samples, n_detectors)
            A batch of channeled inputs, n_detectors (n_channels) should be last.
        training : bool, optional
            Whether the forward pass is called in training or in prediction 
            mode. Default is None.

        Returns
        -------
        tf.Tensor, shape = (None, n_samples, n_detectors)
            The corresponding batch of channeled inputs.
        """

        x = data
        if training:
            x = tf.cond(tf.random.uniform(()) < self.p, 
                        partial(self._permute_channels, data = x),
                        lambda: tf.cast(x, self.dtype))
        return x


    def _permute_channels(
            self,
            data: tf.Tensor
        ) -> tf.Tensor:
        """
        This funtion applies a random permutations of the channels dimension 
        assuming channels last format.

        Parameters
        ----------
        data : tf.Tensor
            Input data.

        Returns
        -------
        tf.Tensor
            Permuted output data.
        """
    
        x = tf.cast(data, self.dtype)
        perm = tf.range(data.get_shape()[-1])
        perm = tf.random.shuffle(perm)
        return tf.gather(x, perm, axis = -1)


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
            "p" : self.p
        }
        
        config.update(super(PermuteChannel, self).get_config())
        return config


##############################################################################


class GaussianNoise(tf.keras.layers.Layer):
    """
    Layer that adds Gaussian noise of 0 mean and specified standard deviation 
    to a signal.
    """

    def __init__(
            self, 
            p: float = 0.1,
            stddev: float = .25,
            **kwargs
        ) -> None:
        """
        Function to initialise the object.

        Parameters
        ----------
        p : float, optional
            Probability of adding noise. The default 0.1.
        """
    
        super(GaussianNoise, self).__init__(**kwargs)
        self.p = p
        self.stddev = stddev


    def build(
            self, 
            input_shape: Tuple[int, int]
        ) -> None:
        """
        Function to build the graph of the layer. Adds trainable and non-
        trainable parameters if any.

        Parameters
        ----------
        input_shape : Tuple[int, int]
            Shape of the input to the layer.
        """

        super(GaussianNoise, self).build(input_shape)


    def call(
            self, 
            data: tf.Tensor,
            training: bool = None
        ) -> tf.Tensor:
        """
        Forward pass of the layer (requires channels to be last).

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_samples, n_detectors)
            A batch of channeled inputs, n_detectors (n_channels) should be last.
        training : bool, optional
            Whether the forward pass is called in training or in prediction 
            mode. Default is None.

        Returns
        -------
        tf.Tensor, shape = (None, n_samples, n_detectors)
            The corresponding batch of channeled inputs.
        """

        x = data
        if training:
            x = tf.cond(tf.random.uniform(()) < self.p, 
                        partial(self._add_noise, data = x),
                        lambda: tf.cast(x, self.dtype))
        return x


    def _add_noise(
            self,
            data: tf.Tensor
        ) -> tf.Tensor:
        """
        This funtion adds random Gaussian noise with 0 mean to the input signal.

        Parameters
        ----------
        data : tf.Tensor
            Input data.

        Returns
        -------
        tf.Tensor
            Output data with noise.
        """

        x = tf.cast(data, self.dtype)
        noise = tf.random.normal(tf.shape(x), stddev = self.stddev, 
                                 dtype = self.dtype)
        return x + noise


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
            "p" : self.p,
            "stddev" : self.stddev
        }
        
        config.update(super(GaussianNoise, self).get_config())
        return config


##############################################################################


class SpectralMask(tf.keras.layers.Layer):
    """
    Layer that applies spectral masks to an input spectrogram. Not usable with TPU.
    """

    def __init__(
            self, 
            p: float = 0.2,
            n_max_mask_t: int = 2,
            w_mask_t: Tuple[int, int] = (5, 10),
            n_max_mask_f: int = 2,
            w_mask_f: Tuple[int, int] = (5, 10),
            **kwargs
        ) -> None:
        """
        Function to initialise the object.

        Parameters
        ----------
        p : float, optional
            Probability of applying a spectral mask. The default 0.2.
        n_max_mask_t : int, optional
            Maximum number of masks in time dimension. The default is 2.
        n_max_mask_f : int, optional
            Maximum number of masks in frequency dimension. The default is 2.
        """
    
        super(SpectralMask, self).__init__(**kwargs)
        self.p = p
        self.n_max_mask_t = tf.math.maximum(n_max_mask_t, 0)
        self.w_min_mask_t = tf.math.maximum(w_mask_t[0], 1)
        self.w_max_mask_t = tf.math.maximum(w_mask_t[-1], 1)
        self.n_max_mask_f = tf.math.maximum(n_max_mask_f, 0)
        self.w_min_mask_f = tf.math.maximum(w_mask_f[0], 1)
        self.w_max_mask_f = tf.math.maximum(w_mask_f[-1], 1)


    def build(
            self, 
            input_shape: Tuple[int, int, int]
        ) -> None:
        """
        Function to build the graph of the layer. Adds trainable and non-
        trainable parameters if any.

        Parameters
        ----------
        input_shape : Tuple[int, int, int]
            Shape of the input to the layer.
        """

        super(SpectralMask, self).build(input_shape)



    def call(
            self, 
            data: tf.Tensor,
            training: bool = None
        ) -> tf.Tensor:
        """
        Forward pass of the layer (requires channels to be last).

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_time, n_freq, n_detectors)
            A batch of channeled inputs, n_detectors (n_channels) should be last.
        training : bool, optional
            Whether the forward pass is called in training or in prediction 
            mode. Default is None.

        Returns
        -------
        tf.Tensor, shape = (None, n_time, n_freq, n_detectors)
            The corresponding batch of channeled inputs.
        """

        x = data
        if training:
            x = tf.cond(tf.random.uniform(()) < self.p, 
                        partial(self._apply_all_mask, data = x),
                        lambda: tf.cast(x, self.dtype))
        return x


    def _apply_single_mask_freq(
            self,
            i: int,
            i_max: int,
            data: tf.Tensor,
        ) -> tf.Tensor:
        """
        This funtion applies a single frequency spectral mask to a spectrogram. 
        Assumes batch as the first dimension and channels as the last dimension.
    
        Parameters
        ----------
        i : int
            Counter of the number of masks applied.
        data : tf.Tensor, shape = (None, n_freq, n_time, n_detectors)
            Input data.
    
        Returns
        -------
        tf.Tensor
            Permuted output data.
        """

        w_mask = tf.random.uniform(shape = (), minval = self.w_min_mask_f, 
                                   maxval = self.w_max_mask_f + 1, dtype = tf.int32)
        x = data
        x = tf.cond(i < i_max, partial(_Utilities.freq_mask, param = w_mask, data = x), 
                    partial(_Utilities.freq_mask, param = 0, data = x))
        return i + 1, i_max, x
    

    def _apply_single_mask_time(
            self,
            i: int,
            i_max: int,
            data: tf.Tensor,
        ) -> tf.Tensor:
        """
        This funtion applies a single temporal spectral mask to a spectrogram. 
        Assumes batch as the first dimension and channels as the last dimension.
    
        Parameters
        ----------
        i : int
            Counter of the number of masks applied.
        data : tf.Tensor, shape = (None, n_freq, n_time, n_detectors)
            Input data.
    
        Returns
        -------
        tf.Tensor
            Permuted output data.
        """
    
        w_mask = tf.random.uniform(shape = (), minval = self.w_min_mask_t, 
                                   maxval = self.w_max_mask_t + 1, dtype = tf.int32)
        x = data
        x = tf.cond(i < i_max, partial(_Utilities.time_mask, param = w_mask, data = x), 
                    partial(_Utilities.time_mask, param = 0, data = x))
        return i + 1, i_max, x


    def _apply_all_mask(
            self,
            data: tf.Tensor
        ) -> tf.Tensor:
        """
        This funtion applies all spectral masks to an input spectrogram 
        according to configuration. Assumes batch as the first dimension and 
        channels as the last dimension.

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_freq, n_time, n_detectors)
            Input data.

        Returns
        -------
        tf.Tensor
            Masked output data.
        """
    
        n_mask_t = tf.random.uniform(shape = (), maxval = self.n_max_mask_t + 1, 
                                      dtype = tf.int32)
        n_mask_f = tf.random.uniform(shape = (), maxval = self.n_max_mask_f + 1, 
                                      dtype = tf.int32)

        x = data
        x = tf.cast(x, self.dtype)
        x = tf.while_loop(lambda i, i_max, inp: i < self.n_max_mask_t, 
                  self._apply_single_mask_time, (0, n_mask_t, x),
                  maximum_iterations = self.n_max_mask_t)[-1]
        x = tf.while_loop(lambda i, i_max, inp: i < self.n_max_mask_f, 
                  self._apply_single_mask_freq, (0, n_mask_f, x),
                  maximum_iterations = self.n_max_mask_f)[-1]
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
            "p" : self.p,
            "n_max_mask_t" : self.n_max_mask_t,
            "w_mask_t" : self.w_mask_t,
            "n_max_mask_f" : self.n_max_mask_t,
            "w_mask_f" : self.w_mask_t
        }
        
        config.update(super(SpectralMask, self).get_config())
        return config


##############################################################################


class TimeMask(tf.keras.layers.Layer):
    """
    Layer that applies a single time mask to an input spectrogram.
    """

    def __init__(
            self, 
            p: float = 0.2,
            w_mask: Tuple[int, int] = (5, 10),
            **kwargs
        ) -> None:
        """
        Function to initialise the object.

        Parameters
        ----------
        p : float, optional
            Probability of applying a spectral mask. The default 0.2.
        w_mask : Tuple[int, int], optional
            Minimum and maximum width of the mask in pixels. The default is (5, 10).
        """
    
        super(TimeMask, self).__init__(**kwargs)
        self.p = tf.constant(p, self.dtype)
        self.w_min_mask = tf.math.maximum(w_mask[0], 1)
        self.w_max_mask = tf.math.maximum(w_mask[-1], 1)


    def build(
            self, 
            input_shape: Tuple[int, int, int]
        ) -> None:
        """
        Function to build the graph of the layer. Adds trainable and non-
        trainable parameters if any.

        Parameters
        ----------
        input_shape : Tuple[int, int, int]
            Shape of the input to the layer.
        """

        super(TimeMask, self).build(input_shape)



    def call(
            self, 
            data: tf.Tensor,
            training: bool = None
        ) -> tf.Tensor:
        """
        Forward pass of the layer (requires channels to be last).

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_time, n_freq, n_detectors)
            A batch of channeled inputs, n_detectors (n_channels) should be last.
        training : bool, optional
            Whether the forward pass is called in training or in prediction 
            mode. Default is None.

        Returns
        -------
        tf.Tensor, shape = (None, n_freq, n_time, n_detectors)
            The corresponding batch of channeled inputs.
        """

        x = data
        if training:
            w_mask = tf.random.uniform(shape = (), minval = self.w_min_mask, 
                                       maxval = self.w_max_mask + 1, dtype = tf.int32)
            x = tf.cast(x, self.dtype)
            x = tf.cond(tf.random.uniform(()) < self.p,
                        partial(_Utilities.time_mask, param = w_mask, data = x),
                        partial(_Utilities.time_mask, 
                        param = tf.constant(0, tf.int32), data = x))
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
            "p" : self.p,
            "w_min_mask" : self.w_min_mask,
            "w_max_mask" : self.w_max_mask
        }
        
        config.update(super(TimeMask, self).get_config())
        return config


##############################################################################


class FreqMask(tf.keras.layers.Layer):
    """
    Layer that applies a single frequency mask to an input spectrogram.
    """

    def __init__(
            self, 
            p: float = 0.2,
            w_mask: Tuple[int, int] = (5, 10),
            **kwargs
        ) -> None:
        """
        Function to initialise the object.

        Parameters
        ----------
        p : float, optional
            Probability of applying a spectral mask. The default 0.2.
        w_mask : Tuple[int, int], optional
            Minimum and maximum width of the mask in pixels. The default is (5, 10).
        """
    
        super(FreqMask, self).__init__(**kwargs)
        self.p = p
        self.w_min_mask = tf.math.maximum(w_mask[0], 1)
        self.w_max_mask = tf.math.maximum(w_mask[-1], 1)


    def build(
            self, 
            input_shape: Tuple[int, int, int]
        ) -> None:
        """
        Function to build the graph of the layer. Adds trainable and non-
        trainable parameters if any.

        Parameters
        ----------
        input_shape : Tuple[int, int, int]
            Shape of the input to the layer.
        """

        super(FreqMask, self).build(input_shape)


    def call(
            self, 
            data: tf.Tensor,
            training: bool = None
        ) -> tf.Tensor:
        """
        Forward pass of the layer (requires channels to be last).

        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_freq, n_time, n_detectors)
            A batch of channeled inputs, n_detectors (n_channels) should be last.
        training : bool, optional
            Whether the forward pass is called in training or in prediction 
            mode. Default is None.

        Returns
        -------
        tf.Tensor, shape = (None, n_freq, n_time, n_detectors)
            The corresponding batch of channeled inputs.
        """

        x = data
        if training:
            w_mask = tf.random.uniform((), minval = self.w_min_mask, 
                                       maxval = self.w_max_mask + 1, dtype = tf.int32)
            x = tf.cast(x, self.dtype)
            x = tf.cond(tf.random.uniform(()) < self.p,
                        partial(_Utilities.freq_mask, param = w_mask, data = x),
                        partial(_Utilities.freq_mask, param = tf.constant(0, tf.int32), 
                                data = x))
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
            "p" : self.p,
            "w_min_mask" : self.w_min_mask,
            "w_max_mask" : self.w_max_mask
        }
        
        config.update(super(FreqMask, self).get_config())
        return config


##############################################################################


class _Utilities(object):
    """ 
    Utilities class for augmentations/regularisations
    """

    @staticmethod
    def time_mask(
            data: tf.Tensor, 
            param: tf.Tensor
        ) -> tf.Tensor:
        """
        Apply masking to a spectrogram in the time domain. Assumes batch 
        as the first dimension and channels as the last dimension.
        
        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_freq, n_time, n_detectors)
            Input spectrogram.
        param: int
            Parameter of time masking indicative of width.
        
        Returns
        -------
        tf.Tensor, shape = (None, n_freq, n_time, n_detectors)
            Masked spectrogram.
        """
        _, freq_max, time_max, _ = data.get_shape()

        t0 = tf.random.uniform((), maxval = time_max - param, dtype = tf.int32)

        indices = tf.reshape(tf.range(time_max), (1, -1))
        condition = tf.math.logical_and(tf.math.greater_equal(indices, t0), 
                                       tf.math.less(indices, t0 + param))

        mask = tf.ones([freq_max, time_max], dtype = data.dtype)
        mask = tf.where(condition, tf.cast(0., data.dtype), mask)
        mask = tf.expand_dims(tf.expand_dims(mask, axis = 0), axis = -1)
        return data * mask


    @staticmethod
    def freq_mask(
            data: tf.Tensor, 
            param: tf.Tensor
        ) -> tf.Tensor:
        """
        Apply masking to a spectrogram in the frequency domain. Assumes batch 
        as the first dimension and channels as the last dimension.
    
        Parameters
        ----------
        data : tf.Tensor, shape = (None, n_freq, n_time, n_detectors)
            Input spectrogram.
        param: tf.Tensor, shape = ()
            Parameter of frequency masking indicative of width.
    
        Returns
        -------
        tf.Tensor, shape = (None, n_freq, n_time, n_detectors)
            Masked spectrogram.
        """
    
        _, freq_max, time_max, _ = data.get_shape()
    
        f0 = tf.random.uniform((), maxval = time_max - param, dtype = tf.int32)
    
        indices = tf.reshape(tf.range(freq_max), (-1, 1))
        condition = tf.math.logical_and(tf.math.greater_equal(indices, f0), 
                                       tf.math.less(indices, f0 + param))
    
        mask = tf.ones([freq_max, time_max], dtype = data.dtype)
        mask = tf.where(condition, tf.cast(0., data.dtype), mask)
        mask = tf.expand_dims(tf.expand_dims(mask, axis = 0), axis = -1)
        return data * mask

    
##############################################################################