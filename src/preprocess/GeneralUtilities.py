# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:51:33 2021

@author: salva
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


##############################################################################

class GeneralUtilities(object):
    """
    General utilities class
    """
    
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
    

##############################################################################