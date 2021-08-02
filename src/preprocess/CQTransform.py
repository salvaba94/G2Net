# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 15:12:40 2021

@author: salva
"""


import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Mapping
from nnAudio.Spectrogram import CQT1992v2
from .GeneralUtilities import GeneralUtilities


##############################################################################

class CQTransform(object):
    """
    Class to compute Constant Q-Transforms and manipulate them, 
    create new preprocessed datasets, etc.
    """

    def __init__(
            self,
            dataframe: pd.DataFrame,
            datadir: Path,
            sample_rate: float, 
            hop_length: float,
            scale: float = 2e-20,
            f_min: float = 0.,
            f_max: float = None
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
        hop_length : float
            Number of samples between successive frames.
        f_min : float
            Minimum frequency in the Mel filter bank [Hz]. Default is 0.
        f_max : float, optional
            Maximum frequency in the Mel filter bank [Hz]. Default is None.

        Returns
        -------
        None
        """
        self.dataframe = dataframe.copy()
        self.datadir = datadir
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.cqt = CQT1992v2(sr = self.sample_rate, 
                             hop_length = self.hop_length, 
                             fmin = self.f_min,
                             fmax = self.f_max)


    def compute_spectrogram(
            self,
            idx: int
        ) -> np.ndarray:
        """
        Function to compute the transform for the specified index.

        Parameters
        ----------
        idx : int
            Index in of the sample used to compute the spectrogram.

        Returns
        -------
        np.ndarray, shape = (n_freq, n_time, n_detectors)
            The corresponding batch of constant Q-transforms
        """
        waveform = GeneralUtilities.get_sample(self.dataframe, 
                                               self.datadir, 
                                               idx, trans = False, 
                                               target = False)
        sc_waveform = waveform / waveform.max()
        print(sc_waveform.shape)

        cqt = self.cqt(torch.from_numpy(sc_waveform).float()).numpy()
        cqt = np.transpose(cqt, (1, 2, 0))
        cqt = GeneralUtilities.scale_linearly(cqt)
        return cqt


    def generate_dataset(
            self,
            n_processes: int,
            destdir: Path,
            dtype: type = np.float32,
        ) -> None:
        """
        Function to generate the a full spectrogram dataset.

        Parameters
        ----------
        n_processes : int
            Dataframe with the indeces of the samples.
        destdir : Path
            Destination directory.
        dtype : type
            Type of the data to store. Default is np.float16.

        Returns
        -------
        None
        """
        destdir.mkdir(parents = True, exist_ok = True)
        n_cpus = np.maximum(n_processes, 0)
        n_cpus = np.minimum(n_cpus, mp.cpu_count())

        with mp.Pool(n_processes) as pool:
            gs = partial(self.generate_spectrogram, destdir = destdir, 
                         dtype = dtype)
            pool.map(gs, self.dataframe.index)


    def generate_spectrogram(
            self,
            idx: int,
            destdir: Path,
            dtype: type = np.float16,
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
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "f_min": self.f_min,
            "f_max": self.f_max,
        }
        return config

##############################################################################