# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 21:54:07 2021

@author: salva
"""

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display


##############################################################################


class PlottingUtilities(object):
    """
    Plotting utilities class
    """

    time_tag = "Time [s]"
    freq_tag = "Frequency [Hz]"
    mag_tag = "Strain [-]"
    detector = ("LIGO Hanford", "LIGO Livingston", "Virgo")


    @classmethod
    def plot_wave(
            cls,
            waveforms: np.ndarray, 
            timespan: float = 2.
        ) -> None:
        """
        Function to plot waves from the 3 detectors.

        Parameters
        ----------
        waveforms : np.ndarray, shape = (n_samples, n_detectors)
            Waveform data to plot.
        timespan : float, optional
            Time span of the waveforms [s]. The default is 2.

        Returns
        -------
        None
        """
        if waveforms.shape[-1] != 3:
            raise ValueError("Function expects exactly data for 3 detectors")

        time = np.linspace(0., timespan, waveforms.shape[0]) [:, np.newaxis]
        min_val, max_val = waveforms.min(), waveforms.max()
        dataframe = pd.DataFrame(data = np.hstack((time, waveforms)), 
                                 columns = [cls.time_tag, 
                                            cls.mag_tag + " " + cls.detector[0], 
                                            cls.mag_tag + " " + cls.detector[1], 
                                            cls.mag_tag + " " + cls.detector[2]])

        plt.style.use('seaborn')
        fig, axes = plt.subplots(3, 1, figsize = (15, 10))
        for i in range(len(cls.detector)):
            sns.lineplot(data = dataframe, x = cls.time_tag, 
                         y = cls.mag_tag + " " + cls.detector[i], ax = axes[i])
            axes[i].legend([cls.detector[i]])
            axes[i].set_ylabel(cls.mag_tag)
            axes[i].set_ylim(min_val, max_val)
    

    @classmethod
    def plot_spectrogram(
            cls,
            spectrogram: np.ndarray,
            sr: float,
            **kwargs
        ) -> None:
        """
        Function to plot a spectrogram

        Parameters
        ----------
        spectrogram : np.ndarray, shape (n_width, n_height, n_detectors)
            Spectrogram data.
        sr : float
            Sampling rate [Hz].

        Returns
        -------
        None
        """
        if spectrogram.shape[-1] != 3:
            raise ValueError("Function expects exactly data for 3 detectors")

        plt.style.use('seaborn')
        fig, axes = plt.subplots(1, 3, figsize = (15, 5))
        for i in range(len(cls.detector)):
            librosa.display.specshow(data = spectrogram[..., i], 
                                     ax = axes[i], **kwargs)
            axes[i].set_xlabel(cls.time_tag)
            axes[i].set_ylabel(cls.freq_tag)
            axes[i].set_title(cls.detector[i])
      
        
      
    @staticmethod
    def plot_count(
            dataframe: pd.DataFrame
        ) -> None:
        """
        Function to display a barplot with the positive and negative examples.

        Parameters
        ----------
        dataframe : pd.DataFrame, columns = (id, target)
            Data with the id and targets of the set.

        Returns
        -------
        None
        """

        plt.style.use('seaborn')
        fig, axes = plt.subplots(1, 1, figsize = (8, 4))
        sns.countplot(data = dataframe, x = "target")
        axes.set_xlabel("Targets")
        axes.set_ylabel("Count [#]")
        
##############################################################################
    