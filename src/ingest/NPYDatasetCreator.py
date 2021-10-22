# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 10:51:16 2021

@author: salva
"""

import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from pathlib import Path
from typing import Tuple
from functools import partial


##############################################################################

class NPYDatasetCreator(object):
    """
    Class to aid in the creation of a preprocessed dataset in npy format.
    """

    def __init__(
            self,
            dataframe: pd.DataFrame,
            datadir: Path,
            trans: bool = False,
            data_stats: Tuple[float, float] = None,
            raw_dir: bool = False,
            ext_in: str = ".npy",
  
        ) -> None:
        """
        Function to initialise the object.
        
        Parameters
        ----------
        dataframe : pd.DataFrame, columns = (id, targets)
            Dataframe with the indeces of the samples.
        datadir : Path
            Data directory.
        trans : bool, optional
            Whether the to transpose the data before storing. The default is 
            False.
        data_stats : Tuple[float, float], optional
            If provided, these are used to standardise the input data. It 
            contains mean and standard deviation in this order. The default is None.
        raw_dir : bool
            Whether the folder should be treated as a raw data folder directory.
            The default is False.
        ext_in : str, optional
            Extension of the input files. The default is ".npy".
        """

        self.df = dataframe.copy()
        self.datadir = datadir
        self.data_stats = data_stats
        self.trans = trans

        if raw_dir:
            self.df["path"] = str(datadir) + os.sep + dataframe["id"].apply(
                lambda x: x[0]) + os.sep + dataframe["id"].apply(
                lambda x: x[1]) + os.sep + dataframe["id"].apply(
                lambda x: x[2]) + os.sep + dataframe["id"].astype(str) + ext_in
        else:
            self.df["path"] = str(datadir) + os.sep + dataframe["id"].astype(str) + ext_in


    def _preprocess_example(
            self, 
            idx,
            dtype: type = np.float16
        ) -> np.ndarray:
        """
        Method to preprocess a single example.
        
        Parameters
        ----------
        idx : int
            ID of the example to preprocess.
        dtype : type, optional
            Data type to which the example should be casted to. The default 
            is np.float16.
        """

        data = np.load(self.df["path"][idx])
        data = data.T if self.trans else data
        if self.data_stats is not None:
            data = (data - self.data_stats[0]) / self.data_stats[-1] 
        data = data.astype(dtype)
        return data


    def _save_example(
            self,
            idx: int,
            destdir: Path,
            dtype: type = np.float16,
            ext_out: str = ".npy"
        ) -> None:
        """
        Method to preprocess a single example and save it.
        
        Parameters
        ----------
        idx : int
            ID of the example to preprocess and save.
        destdir : Path
            Destination directory.
        dtype : type, optional
            Data type to which the example should be saved. The default 
            is np.float16.
        ext_out : str, optional
            Extension of the output files. The default is ".npy".
        """
        filename = destdir.joinpath(self.df["id"][idx] + ext_out)
        data = self._preprocess_example(idx, dtype = dtype)
        np.save(filename, data)


    def create_dataset(
            self,
            destdir: Path,
            dtype: type = np.float16,
            ext_out: str = ".npy",
            n_processes: int = 1
        ) -> None:
        """
        Method to create a full preprocessed dataset and write it to npy files.
        
        Parameters
        ----------
        destdir : Path
            Destination directory.
        dtype : type, optional
            Data type to which the examples should be saved. The default 
            is np.float16.
        ext_out : str, optional
            Extension of the output files. The default is ".npy".
        n_processes: int, optional
            Number of processes reading, preprocessing and saving files. The 
            default is 1.
        """

        n_cpus = np.maximum(n_processes, 0)
        n_cpus = np.minimum(n_cpus, mp.cpu_count())

        destdir.mkdir(parents = True, exist_ok = True)

        with mp.Pool(n_cpus) as pool:
            writer = partial(self._save_example, destdir = destdir, 
                         dtype = dtype, ext_out = ext_out)
            pool.map(writer, self.df.index)

