# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 18:54:23 2021

@author: salva
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path, os
from typing import Tuple
from tensorflow.data.experimental import AUTOTUNE

from .TFRDatasetCreator import TFRDatasetCreator



##############################################################################

class DatasetGeneratorTF(object):
    """
    Class to aid in the creation of dataset pipelines using Tensorflow
    """


    def __init__(
            self,
            dataframe: pd.DataFrame,
            datadir: Path,
            batch_size: int = 64,
            dtype: type = tf.float32,
            wave_stats: Tuple[float, float] = None,
            raw_dir: bool = False,
            trans: bool = False,
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
            Extension of the files. The default is ".tfrec".

        Returns
        -------
        None
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
        example_data = tf.cast(np.load(filename), self.dtype)
        return example_data


    def _get_dataset_from_npy(
            self,
            shuffle: bool = True,
            buffer_size: int = 1000,
            repeat: bool = True,
            target: bool = True
        ) -> tf.data.Dataset:
        """
        Function to get the dataset pipeline from npy files

        Parameters
        ----------
        shuffle_buffer : int, optional
            Shuffle buffer size. If None, no shuffle will be applied. The default 
            is 3200.

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

        ds = ds.shuffle(buffer_size) if shuffle else ds
        ds = ds.repeat() if repeat else ds

        ds = ds.batch(self.batch_size)
        return ds.prefetch(AUTOTUNE)


    def _get_dataset_from_tfrec(
            self,
            ignore_order: bool = True,
            shuffle: bool = True,
            buffer_size: int = 1000,
            repeat: bool = True,
            target: bool = True,
            identify: bool = False
        ) -> tf.data.Dataset:
        """
        Function to get the dataset pipeline from TFRecords.

        Parameters
        ----------
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

        ds = tf.data.TFRecordDataset(self.df["path"], num_parallel_reads = AUTOTUNE)
        if ignore_order:
            ignore_order = tf.data.Options()
            ignore_order.experimental_deterministic = False
            ds = ds.with_options(ignore_order)

        ds = ds.map(lambda x: TFRDatasetCreator.deserialize_example(
            x, dtype = self.dtype, target = target, identify = identify), 
            num_parallel_calls = AUTOTUNE)

        ds = ds.shuffle(buffer_size) if shuffle else ds
        ds = ds.repeat() if repeat else ds
        
        ds = ds.batch(self.batch_size)
        return ds.prefetch(AUTOTUNE)


    def get_dataset(
            self,
            tfrec: bool = True,
            ignore_order: bool = True,
            shuffle: bool = True,
            buffer_size: int = 3200,
            repeat: bool = True,
            target: bool = True,
            identify: bool = False
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
        identify : bool, optional
            Whether to append the identity to the predictions. Only for TFRecords mode.
            Default is false.

        Returns
        -------
        tf.data.Dataset
            Dataset pipeline.
        """
        
        ret_val = self._get_dataset_from_tfrec(ignore_order, shuffle, buffer_size,
            repeat, target, identify) if tfrec else self._get_dataset_from_npy(
            shuffle, buffer_size, repeat, target)
        return ret_val
            

##############################################################################
