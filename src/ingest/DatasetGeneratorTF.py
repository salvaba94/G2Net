# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 18:54:23 2021

@author: salva
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path, os
from tensorflow.data.experimental import AUTOTUNE

from .TFRDatasetCreator import TFRDatasetCreator
from config import Config



##############################################################################

class DatasetGeneratorTF(object):
    """
    Class to aid in the creation of dataset pipelines using tensorflow.
    """

    def __init__(
            self,
            dataframe: pd.DataFrame,
            datadir: Path,
            batch_size: int = 64,
            dtype: type = tf.float32,
            raw_dir: bool = False,
            ext: str = ".tfrec"
        ) -> None:
        """
        Function to initialise the object.

        Parameters
        ----------
        dataframe : pd.DataFrame, columns = (id, targets)
            Dataframe with the indeces of the samples.
        datadir : Path
            Data directory.
        batch_size : int, optional
            Batch size. The default is 32.
        dtype : type, optional
            Data type to use. The default is np.float16.
        raw_dir : bool
            Whether the folder should be treated as a raw data folder directory.
            The default is False.
        ext : str, optional
            Extension of the files. The default is ".tfrec".
        """

        self.df = dataframe.copy()
        self.datadir = datadir
        self.batch_size = batch_size
        self.dtype = dtype

        if raw_dir:
            self.df["path"] = str(datadir) + os.sep + dataframe["id"].apply(
                lambda x: x[0]) + os.sep + dataframe["id"].apply(
                lambda x: x[1]) + os.sep + dataframe["id"].apply(
                lambda x: x[2]) + os.sep + dataframe["id"].astype(str) + ext
        else:
            self.df["path"] = str(datadir) + os.sep + dataframe["id"].astype(str) + ext


    def _read_npy(
            self, 
            filename: Path
        ) -> np.ndarray:
        """
        Auxiliary method to read data from npy file.

        Parameters
        ----------
        filename : Path
            Path and name of the example to read.

        Returns
        -------
        np.ndarray
            Data from the file.
        """

        example_data = tf.cast(np.load(filename), self.dtype)
        return example_data


    def _get_dataset_from_npy(
            self,
            shuffle: bool = True,
            buffer_size: int = 1024,
            repeat: bool = True,
            target: bool = True
        ) -> tf.data.Dataset:
        """
        Function to get the dataset pipeline from npy files.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to add shuffle to the pipeline or not. The default is True.
        buffer_size : int, optional
            Shuffle buffer size. The default is 1024.
        repeat : bool, optional
            Whether to repeat the dataset or not. The default is True
        target : bool, optional
            Whether to add the label or not. The default is True.

        Returns
        -------
        tf.data.Dataset
            Dataset pipeline.
        """

        feature_ds = tf.data.Dataset.from_tensor_slices(self.df["path"])
        feature_ds = feature_ds.map(lambda x: tf.numpy_function(
            self._read_npy, [x], self.dtype), 
            num_parallel_calls = AUTOTUNE)

        if target:
            label_ds = tf.data.Dataset.from_tensor_slices(self.df["target"])
            ds = tf.data.Dataset.zip((feature_ds, label_ds))
        else:
            ds = feature_ds

        if shuffle:
            ds = ds.shuffle(buffer_size)
        
        if repeat:
            ds = ds.repeat()

        ds = ds.batch(self.batch_size)
        return ds.prefetch(AUTOTUNE)


    def _get_dataset_from_tfrec(
            self,
            shuffle: bool = True,
            buffer_size: int = 1024,
            ordered: bool = False,
            repeat: bool = True,
            target: bool = True
        ) -> tf.data.Dataset:
        """
        Function to get the dataset pipeline from tensorflow records.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to add shuffle to the pipeline or not. The default is True.
        buffer_size : int, optional
            Shuffle buffer size. The default is 1024.
        ordered: bool, optional
            Indicate whether the order matters when reading. The default is False.
        repeat : bool, optional
            Whether to repeat the dataset or not. The default is True.
        target : bool, optional
            Whether to add the label or not. The default is True.
        
        Returns
        -------
        tf.data.Dataset
            Dataset pipeline.
        """

        ds = tf.data.TFRecordDataset(self.df["path"], num_parallel_reads = AUTOTUNE)

        if not ordered:
            ignore_order = tf.data.Options()
            ignore_order.experimental_deterministic = False
            ds = ds.with_options(ignore_order)

        ds = ds.map(lambda x: TFRDatasetCreator.deserialize_example(
            x, dtype = self.dtype, target = target, shape = (Config.N_SAMPLES, 
            Config.N_DETECT)), num_parallel_calls = AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(buffer_size)

        if repeat:
            ds = ds.repeat()
        
        ds = ds.batch(self.batch_size, drop_remainder = target)
        return ds.prefetch(AUTOTUNE)


    def get_dataset(
            self,
            tfrec: bool = True, 
            shuffle: bool = True,
            buffer_size: int = 1024,
            ordered: bool = False,
            repeat: bool = True,
            target: bool = True
        ) -> tf.data.Dataset:
        """
        Function to get the dataset pipeline.

        Parameters
        ----------
        tfrec : bool, optional
            Whether TFRecord should be read or not. The default is True.
        shuffle_buffer : int, optional
            Shuffle buffer size. If None, no shuffle will be applied. The default 
            is 3200.
        ignore_order : bool, optional
            Dataframe with the indeces of the samples. The default is True.

        Returns
        -------
        tf.data.Dataset
            Dataset pipeline.
        """
        
        ret_val = self._get_dataset_from_tfrec(shuffle, buffer_size, ordered,
            repeat, target) if tfrec else self._get_dataset_from_npy(
            shuffle, buffer_size, repeat, target)
        return ret_val
            

##############################################################################
