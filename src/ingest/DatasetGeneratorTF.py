# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 18:54:23 2021

@author: salva
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path, os
from typing import List
from tensorflow.data.experimental import AUTOTUNE



##############################################################################

class DatasetGeneratorTF(object):
    """
    Class to aid in the creation of dataset pipelines using Tensorflow
    """

    def __init__(
            self,
            dataframe: pd.DataFrame,
            datadir: Path,
            shuffle: bool = True,
            shuffle_buffer: int = 1000,
            batch_size: int = 32,
            dtype: type = tf.float16,
            raw_dir: bool = False,
            image_size: List[int] = [128, 128],
            target: bool = True,
            ext: str = ".npy"
        ) -> None:
        """
        Function to initialise the object.

        Parameters
        ----------
        dataframe : pd.DataFrame, columns = (id, targets)
            Dataframe with the indeces of the samples.
        datadir : Path
            Data directory.
        shuffle_buffer : int, optional
            Shuffle buffer. The default is 50000.
        batch_size : int, optional
            Batch size. The default is 32.
        dtype : type, optional
            Data type to use. The default is np.float16.
        raw_dir : bool
            Whether the folder should be treated as a raw data folder directory.
            The default is False.
        target : bool, optional
            Whether the target of the sample should be provided. The default is 
            True.
        ext : str, optional
            Extension of the files. The default is ".npy".

        Returns
        -------
        None
        """

        self.df = dataframe.copy()
        self.datadir = datadir
        self.batch_size = batch_size
        self.dtype = dtype
        self.target = target
        self.image_size = image_size
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer

        if raw_dir:
            self.df["paths"] = str(datadir) + os.sep + dataframe["id"].apply(
                lambda x: x[0]) + os.sep + dataframe["id"].apply(
                lambda x: x[1]) + os.sep + dataframe["id"].apply(
                lambda x: x[2]) + os.sep + dataframe["id"].astype(str) + ext
        else:
            self.df["paths"] = str(datadir) + os.sep + dataframe["id"].astype(str) + ext


    def _read_npy(
            self, 
            filename
        ) -> np.ndarray:
        example_data = np.load(filename)
        example_data = tf.cast(tf.image.resize(example_data, self.image_size), 
                               dtype = self.dtype)
        return example_data
    

    def get_dataset(
            self
        ) -> tf.data.Dataset:
        """
        Function to get the dataset pipeline.

        Returns
        -------
        tf.data.Dataset
            Dataset pipeline.
        """
        feature_ds = tf.data.Dataset.from_tensor_slices(self.df["paths"])
        feature_ds = feature_ds.map(lambda x: tf.numpy_function(
            self._read_npy, [x], self.dtype), 
            num_parallel_calls = AUTOTUNE)

        if self.target:
            label_ds = tf.data.Dataset.from_tensor_slices(self.df["target"])
            ds = tf.data.Dataset.zip((feature_ds, label_ds))
        else:
            ds = feature_ds

        if self.shuffle:
            ds = ds.shuffle(self.shuffle_buffer)
        ds = ds.batch(self.batch_size)
        return ds.prefetch(AUTOTUNE)
        
##############################################################################