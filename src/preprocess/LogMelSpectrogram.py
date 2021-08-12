# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:03:31 2021

@author: salva
"""

import librosa
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Mapping, Tuple
from .GeneralUtilities import GeneralUtilities


##############################################################################

class LogMelSpectrogram(object):
    """
    Class to compute log-magnitude mel-spectrograms and manipulate them, 
    create new preprocessed datasets, etc.
    """

    def __init__(
            self,
            dataframe: pd.DataFrame,
            datadir: Path,
            n_processes: int,
            sample_rate: float, 
            n_mels: float, 
            n_fft: float, 
            hop_length: float,
            f_band: Tuple[float, float] = (0., None)
        ) -> None:
        """
        Function to initialise the object.

        Parameters
        ----------
        dataframe : pd.DataFrame, columns = (id, targets)
            Dataframe with the indeces of the samples.
        datadir : Path
            Data directory.
        sample_rate : float
            Sampling rate [Hz].
        n_mels : float
            Number of Mel bands to generate.
        n_fft : float
            Length of the FFT window.
        hop_length : float
            Number of samples between successive frames.

        f_band : float
            Minimum and maximum values to scale the signal before CQT [-]. 
            Default is None.

        Returns
        -------
        None
        """
        self.dataframe = dataframe.copy()
        self.datadir = datadir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_band = f_band

        n_cpus = np.maximum(n_processes, 0)
        self.n_cpus = np.minimum(n_cpus, mp.cpu_count())

    def compute_spectrogram(
            self,
            idx: int
        ) -> np.ndarray:
        """
        Function to compute the spectrogram for the specified index.

        Parameters
        ----------
        idx : int
            Index in of the sample used to compute the spectrogram.

        Returns
        -------
        np.ndarray, shape = (n_freq, n_time, n_detectors)
            The corresponding batch of log-scaled Mel-spectrograms.
        """
        full_mel_spec = np.array([])
        waveform = GeneralUtilities.get_sample(self.dataframe, 
                                               self.datadir, 
                                               idx, trans = True, 
                                               target = False)
        sc_waveform = waveform / waveform.max()

        for i in range(waveform.shape[-1]):
            mel_spec = librosa.feature.melspectrogram(sc_waveform[..., i], 
                                                      sr = self.sample_rate, 
                                                      n_mels = self.n_mels, 
                                                      n_fft = self.n_fft, 
                                                      hop_length = self.hop_length, 
                                                      fmin = self.f_band[0],
                                                      fmax = self.f_band[-1])
            mel_spec = librosa.power_to_db(mel_spec)
            full_mel_spec = np.dstack((full_mel_spec, mel_spec)) \
                if full_mel_spec.size else mel_spec

        full_mel_spec = GeneralUtilities.scale_linearly(full_mel_spec)

        return full_mel_spec


    def generate_dataset(
            self,
            destdir: Path,
            dtype: type = np.float16,
        ) -> None:
        """
        Function to generate the a full spectrogram dataset.

        Parameters
        ----------
        destdir : Path
            Destination directory.
        dtype : type
            Type of the data to store. Default is np.float16.

        Returns
        -------
        None
        """
        destdir.mkdir(parents = True, exist_ok = True)

        with mp.Pool(self.n_cpus) as pool:
            gs = partial(self.generate_spectrogram, destdir = destdir, 
                         dtype = dtype)
            pool.map(gs, self.dataframe.index)


    def generate_spectrogram(
            self,
            idx: int,
            destdir: Path,
            dtype: type = np.float32,
            ext: str = ".npy"
        ) -> np.ndarray:
        """
        Function to generate a single spectrogram example and save it.

        Parameters
        ----------
        idx : int
            Index in of the sample used to compute the spectrogram.
        destdir : Path
            Destination directory.
        dtype : type
            Type of the data to store. Default is np.float16.
        ext : str, optional
            Extension of the data files. The default is ".npy".

        Returns
        -------
        None
        """
        spec = self.compute_spectrogram(idx).astype(dtype)
        filename = self.dataframe["id"][idx]
        full_filename = destdir.joinpath(filename + ext)
        np.save(full_filename, spec)
        
    
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
            "datadir": self.datadir,
            "dataframe": self.dataframe,
            "n_cpus": self.n_cpus,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "f_band": self.f_band
        }
        return config

##############################################################################