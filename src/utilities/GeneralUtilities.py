# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:51:33 2021

@author: salva
"""

import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from typing import Tuple
import multiprocessing as mp
import tensorflow as tf


##############################################################################

class GeneralUtilities(object):
    """
    General utilities class
    """
    
    safe_term = 1e-20
    
    @staticmethod
    def get_dims(
            datadir: Path, 
            ext: str = ".npy",
            trans: bool = False
        ) -> Tuple[int, int]:
        """
        Function to obtain the shape of the data points in a directory.

        Parameters
        ----------
        datadir : Path
            Directory where the data samples are.
        ext : str, optional
            Extension of the data samples. The default is ".npy".
        trans : bool, optional
            Whether to transpose or not. The default is False.

        Returns
        -------
        Tuple[int, int]
            Tuple containing the shape of the data points.

        """
        example_files_gen = datadir.glob("**/*" + ext)
        for example_file in example_files_gen:
            example_data = np.load(example_file)
            break
        example_data = example_data.T if trans else example_data
        return example_data.shape


    @staticmethod
    def get_sample(
            df: pd.DataFrame, 
            datadir: Path, 
            idx: int, 
            target: bool = True, 
            raw_dir: bool = True,
            trans: bool = False,
            ext: str = ".npy"
        ) -> Tuple[np.ndarray, int]:
        """
        Function to retrieve samples provided an index.

        Parameters
        ----------
        df : pd.DataFrame, columns = (id, targets)
            Data with the id and targets of the set.
        datadir : Path
            Path to data.
        idx : int
            Index of the sample to retrieve.
        target : bool, optional
            Whether the target of the sample should be provided. The default is 
            True.
        raw_dir : bool
            Whether the folder should be treated as a raw data folder directory.
            The default is True.
        trans : bool
            Whether the example data should be transposed or not. The default 
            is False.
        ext : str, optional
            Extension of the data files. The default is ".npy".

        Returns
        -------
        Tuple[np.ndarray, int]
            Retrieved sample, containing target value if specified 
        """
        example_label = df["target"][idx]
        example_id = df["id"][idx]
        filename = datadir.joinpath(example_id[0], example_id[1], example_id[2]) \
            if raw_dir else datadir
        filename = filename.joinpath(example_id + ext)
        example_data = np.load(filename)
        example_data = example_data.T if trans else example_data
        return_val = (example_data, example_label) if target else example_data
        return return_val


    @staticmethod
    def scale_linearly(
            magnitude: np.ndarray, 
            pre_norm: bool = True,
            band_in: Tuple[float, float] = None,
            band_out: Tuple[float, float] = (0., 255.), 
        ) -> np.ndarray:
        """
        Function to scale linearly an array of data.

        Parameters
        ----------
        magnitude : np.ndarray, shape = (n_samples, n_detectors)
            Array of data to scale linearly.
        pre_norm : bool
            Whether to pre-normalise or not. The default is true
        band_in : Tuple[float, float], optional
            Minimum and maximum values for input array. The default is None,
            which means that the minimum and maximum from magnitude array will
            be used 
        band_out : Tuple[float, float] optional
             Minimum and maximum values for output array. The default is (0, 255).

        Returns
        -------
        np.ndarray
            Scaled output array.
        """
        min_max_norm = magnitude
        if pre_norm:
            if band_in is None:
                min_val, max_val = magnitude.min(), magnitude.max()
            else:
                min_val, max_val = band_in

            min_max_norm = (magnitude - min_val) / (max_val - min_val)
        return band_out[0] + min_max_norm * (band_out[-1] - band_out[0]) 


    @classmethod
    def _get_wave_mean(
            cls,
            idx: int,
            sample_df: pd.DataFrame,
            datadir: Path
        ) -> float:
        """
        Function to get the mean of a single sample.

        Parameters
        ----------
        idx : int
            Index in of the example used to compute the mean.
        sample_df : pd.DataFrame
            Sample dataset with the example IDs from which to extract the 
            statistics. Could contain the full dataset or a reduced version.
        datadir : Path
            Directory where the data is stored.

        Returns
        -------
        float
            Mean of the sample
        """
        
        waveform = cls.get_sample(sample_df, datadir, idx, trans = False, 
                                  target = False)
        waveform /= cls.safe_term
        return waveform.mean()
    

    @classmethod
    def _get_wave_sqdiff(
            cls,
            idx: int,
            means: np.ndarray,
            sample_df: pd.DataFrame,
            datadir: Path
        ) -> float:
        """
        Function to get the sum of squared differences of a single sample.

        Parameters
        ----------
        idx : int
            Index in of the sample used to compute the mean.
        means : np.ndarry
            Numpy array with all the means of the different samples.
        sample_df : pd.DataFrame
            Sample dataset with the example IDs from which to extract the 
            statistics. Could contain the full dataset or a reduced version.
        datadir : Path
            Directory where the data is stored.

        Returns
        -------
        float
            Sum of squared differences of the sample.
        """
        
        waveform = cls.get_sample(sample_df, datadir, idx, trans = False, 
                                  target = False)
        waveform /= cls.safe_term
        return ((waveform - means[idx]) ** 2.).sum()
    

    @classmethod
    def _generate_statistics(
            cls,
            filename: Path,
            sample_df: pd.DataFrame,
            datadir: Path,
            n_processes: int = 1,
        ) -> np.ndarray:
        """
        Function to generate the statistics of a full waveform dataset. Aimed 
        to be used for a single dataset and extrapolated to the others.

        Parameters
        ----------
        filename : Path
            Filename where the statistics will be stored.
        sample_df : pd.DataFrame
            Sample dataset with the example IDs from which to extract the 
            statistics. Could contain the full dataset or a reduced version.
        datadir : Path
            Directory where the data is stored.
        n_processes : int, optional
            Number of processes to compute the statistics. A map reduce will 
            be performed. The default is 1.

        Returns
        -------
        np.ndarray, shape = (2,)
            Numpy array containing the overall stats of the full dataset in this 
            order: mean and standard deviation.
        """
        n_cpus = np.maximum(n_processes, 0)
        n_cpus = np.minimum(n_cpus, mp.cpu_count())

        with mp.Pool(n_cpus) as pool:
            get_means = partial(cls._get_wave_mean, sample_df = sample_df, 
                                datadir = datadir)
            means = np.array(pool.map(get_means, sample_df.index))

        with mp.Pool(n_cpus) as pool:
            get_sqdiffs = partial(cls._get_wave_sqdiff, means = means, 
                                  sample_df = sample_df, datadir = datadir)
            sqdiffs = np.array(pool.map(get_sqdiffs, sample_df.index))

        n_points = np.prod(GeneralUtilities.get_dims(datadir))
        n_samples = sample_df.shape[0]
        mean = means.mean() * cls.safe_term
        std = np.sqrt(sqdiffs.sum()) / np.sqrt(n_points) / np.sqrt(n_samples) \
            * cls.safe_term
        stats = np.hstack((mean, std))

        np.save(filename, stats)
        return stats


    @classmethod
    def get_stats(
            cls,
            sample_df: pd.DataFrame,
            datadir: Path,
            n_processes: int = None
        ) -> np.ndarray:
        """
        Function to get the statistics from the dataset, primarily thought for 
        raw data. Statistics include mean and standard deviation and two 
        passes are needed to compute it.

        Parameters
        ----------
        sample_df : pd.DataFrame
            Sample dataset with the example IDs from which to extract the 
            statistics. Could contain the full dataset or a reduced version.
        datadir : Path
            Directory where the data is stored.
        n_processes : int, optional
            Number of processes to compute the statistics. A map reduce will 
            be performed. The default is None.

        Raises
        ------
        ValueError
            Error raised if n_processes is None and a computation is needed.

        Returns
        -------
        np.ndarray, shape = (2,)
            Tuple containing the mean and the standard deviation of the dataset.
        """

        print("Getting data statistics...")
        data_file = datadir.joinpath("..","wave_stats.npy")
        if not data_file.exists() and n_processes is None:
            raise ValueError("Stats file does not exist and needs to be \
                             computed, please provide a number of processes")
        data_stats = np.load(data_file) if data_file.exists() else \
            cls._generate_statistics(data_file, sample_df, datadir, 
                                     n_processes = n_processes)
        print("Done!")
        return data_stats


    @staticmethod
    def broadcast_dim(
            x: tf.Tensor
        ) -> tf.Tensor:
        """
        Auto broadcast input for tensorflow.

        Parameters
        ----------
        x : tf.Tensor
            Broadcasted tensor.

        Raises
        ------
        ValueError
            If input shape is not 1, 2 or 3.

        Returns
        -------
        x : tf.Tensor
            Broadcasted tensor.
        """
        rank = len(x.get_shape().as_list())

        if rank == 1:
            x = tf.expand_dims(tf.expand_dims(x, axis = 0), axis = -1)
        elif rank == 2:
            x = tf.expand_dims(x, axis = -1)
        elif rank == 3:
            pass
        else:
            raise ValueError("Only support input with shape = (n_batch, n_samples) \
                             or shape = (n_samples)")
        return x

##############################################################################