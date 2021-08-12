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
from typing import Mapping, Tuple
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
            n_processes: int,
            sample_rate: float, 
            hop_length: float,
            f_band: Tuple[float, float] = (0., None),
            sample_size: int = 100000
        ) -> None:
        """
        Function to initialise the object.

        Parameters
        ----------
        dataframe : pd.DataFrame, columns = (id, targets)
            Dataframe with the indeces of the samples.
        datadir : Path
            Data directory.
        n_processes : int
            Number of parallel processes for task requiring access to the files.
        sample_rate : float
            Sampling rate [Hz].
        hop_length : float
            Number of samples between successive frames.
        f_band : float, optional
            Minimum and maximum frequency in the CQT filter bank [Hz]. 
            Default is (0., None)
        sample_size : int, optional
            Sample size for determining statistics from the training set.

        Returns
        -------
        None
        """
        self.dataframe = dataframe.copy()
        self.datadir = datadir
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f_band = f_band
        self.cqt = CQT1992v2(sr = self.sample_rate, 
                             hop_length = self.hop_length, 
                             fmin = self.f_band[0],
                             fmax = self.f_band[-1])

        size = np.minimum(sample_size, self.dataframe.shape[0])
        self.sample_df = self.dataframe.sample(n = size).reset_index(drop = True)

        n_cpus = np.maximum(n_processes, 0)
        self.n_cpus = np.minimum(n_cpus, mp.cpu_count())
        
        self.safe_term = 1e-20
        self.perc_range = 0.025

        wave_file = self.datadir.joinpath("..","wave_stats.npy")
        self.wave_stats = np.load(wave_file) if wave_file.exists() else \
            self._generate_statistics(wave_file)
            



    def _get_wave_mean(
            self,
            idx: int
        ) -> float:
        """
        Function to get the mean of a single sample.

        Parameters
        ----------
        idx : int
            Index in of the sample used to compute the mean.

        Returns
        -------
        float
            Mean of the sample
        """
        
        waveform = GeneralUtilities.get_sample(self.sample_df, 
                                               self.datadir, 
                                               idx, trans = False, 
                                               target = False)
        waveform /= self.safe_term
        return waveform.mean()
    

    def _get_wave_sqdiff(
            self,
            idx: int,
            means: np.ndarray
        ) -> float:
        """
        Function to get the sum of squared differences of a single sample.

        Parameters
        ----------
        idx : int
            Index in of the sample used to compute the mean.
        means : np.ndarry
            Numpy array with all the means of the different samples

        Returns
        -------
        float
            Sum of squared differences of the sample
        """
        
        waveform = GeneralUtilities.get_sample(self.sample_df, 
                                               self.datadir, 
                                               idx, trans = False, 
                                               target = False)
        waveform /= self.safe_term
        return ((waveform - means[idx]) ** 2.).sum()


    def _generate_statistics(
            self,
            filename: Path
        ) -> np.ndarray:
        """
        Function to generate the statistics of a full waveform dataset. Aimed 
        to be used for a single dataset and extrapolated to the others.

        Parameters
        ----------
        filename : Path
            Filename where the statistics will be stored.

        Returns
        -------
        nd.array, shape = (2,)
            Numpy array containing the overall stats of the full dataset in this 
            order: mean and standard deviation.
        """

        with mp.Pool(self.n_cpus) as pool:
            means = np.array(pool.map(self._get_wave_mean, self.sample_df.index))

        with mp.Pool(self.n_cpus) as pool:
            get_sqdiffs = partial(self._get_wave_sqdiff, means = means)
            sqdiffs = np.array(pool.map(get_sqdiffs, self.sample_df.index))

        n_points = np.prod(GeneralUtilities.get_dims(self.datadir))
        n_samples = self.sample_df.shape[0]
        mean = means.mean() * self.safe_term
        std = np.sqrt(sqdiffs.sum()) / np.sqrt(n_points) / np.sqrt(n_samples) \
            * self.safe_term
        stats = np.hstack((mean, std))

        np.save(filename, stats)
        return stats


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
        sc_waveform = (waveform - self.wave_stats[0])/self.wave_stats[-1]

        cqt = self.cqt(torch.from_numpy(sc_waveform).float()).numpy()
        cqt = np.transpose(cqt, (1, 2, 0))
        return cqt


    def generate_dataset(
            self,
            destdir: Path,
            dtype: type = np.float32,
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
            spec_generator = partial(self._generate_spectrogram, destdir = destdir, 
                         dtype = dtype)
            spec_stats = np.vstack(pool.map(spec_generator, self.dataframe.index))
            
        spec_file = destdir.joinpath("..", "spec_stats.npy")
        spec_stats = np.load(spec_file) if spec_file.exists() else \
            np.hstack((spec_stats[:,0].min(), spec_stats[:,-1].max()))
        np.save(spec_file, spec_stats)
        
        range_val = spec_stats[-1] - spec_stats[0]
        min_val = spec_stats[0] - self.perc_range * range_val
        max_val = spec_stats[-1] + self.perc_range * range_val
        with mp.Pool(self.n_cpus) as pool:
            spec_scaler = partial(self._scale_spectrogram, destdir = destdir, 
                         dtype = dtype, band_in = (min_val, max_val))
            np.vstack(pool.map(spec_scaler, self.dataframe.index))
            

    def _scale_spectrogram(
            self,
            idx: int,
            destdir: Path,
            band_in: Tuple[float, float],
            dtype: type = np.float16,
            ext: str = ".npy"
        ) -> None:
        """
        Function to scale a single spectrogram that has already been calculated.
        File with spectrogram data needs to exist.

        Parameters
        ----------
        idx : int
            Index in of the sample used to compute the spectrogram.
        destdir : Path
            Destination directory.
        band_in : Tuple[float, float]
            Minimum and maximum values for input array.
        dtype : type
            Type of the data to store. Default is np.float16.
        ext : str, optional
            Extension of the data files. The default is ".npy".

        Returns
        -------
        None
        """
        filename = self.dataframe["id"][idx]
        full_filename = destdir.joinpath(filename + ext)

        spec = np.load(full_filename)
        sc_spec = GeneralUtilities.scale_linearly(spec, band_in = band_in).astype(dtype)
        sc_spec = np.clip(sc_spec, 0., 255.)

        full_filename.unlink()
        np.save(full_filename, sc_spec)



    def _generate_spectrogram(
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
        np.ndarray
            Array containing some statistics from the spectrogram in this order:
            minimum, maximum.
        """
        spec = self.compute_spectrogram(idx).astype(dtype)
        filename = self.dataframe["id"][idx]
        full_filename = destdir.joinpath(filename + ext)
        np.save(full_filename, spec)
        return np.array([spec.min(), spec.max()])
        
    
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