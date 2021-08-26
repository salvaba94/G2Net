# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 11:32:58 2021

@author: salva
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import multiprocessing as mp
from pathlib import Path
from typing import Union, Tuple
from functools import partial


##############################################################################

class TFRDatasetCreator(object):
    """
    Class to aid in the creation of tensorflow records datasets
    """

    def __init__(
            self,
            dataframe: pd.DataFrame,
            datadir: Path,
            trans: bool = False,
            target: bool = False,
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
        target : bool, optional
            Whether the target of the sample should be provided. The default is 
            False.
        shuffle_buffer : int, optional
            Shuffle buffer. The default is 50000.
        dtype : type, optional
            Data type to use. The default is np.float16.
        raw_dir : bool
            Whether the folder should be treated as a raw data folder directory.
            The default is False.
        ext_in : str, optional
            Extension of the input files. The default is ".npy".

        Returns
        -------
        None
        """

        self.df = dataframe.copy()
        self.datadir = datadir
        self.target = target
        self.data_stats = data_stats
        self.trans = trans

        if raw_dir:
            self.df["path"] = str(datadir) + os.sep + dataframe["id"].apply(
                lambda x: x[0]) + os.sep + dataframe["id"].apply(
                lambda x: x[1]) + os.sep + dataframe["id"].apply(
                lambda x: x[2]) + os.sep + dataframe["id"].astype(str) + ext_in
        else:
            self.df["path"] = str(datadir) + os.sep + dataframe["id"].astype(str) + ext_in


    @staticmethod
    def _bytes_feature(
            value: Union[np.ndarray, tf.Tensor, str]
        ) -> tf.train.Feature:
        """
        Function to convert an array, string or tensor to a list of bytes feature

        Parameters
        ----------
        value : np.ndarray
            Input value to convert to a list of bytes feature.

        Returns
        -------
        tf.train.Feature
            The converted feature.
        """

        value = value.numpy() if isinstance(value, type(tf.constant(0))) else value
        value = value.encode() if isinstance(value, str) else value

        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))


    @classmethod
    def _array_feature(
            cls,
            value: np.ndarray,
            dtype: type = tf.float32
        ) -> tf.train.Feature:
        
        tensor = tf.convert_to_tensor(value, dtype = dtype)
        return cls._bytes_feature(tf.io.serialize_tensor(tensor))


    @staticmethod
    def _int_feature(
            value: Union[int, bool]
        ) -> tf.train.Feature:
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))


    def _serialize_example(
            self, 
            idx,
            dtype: type = tf.float16
        ) -> str:

        data = np.load(self.df["path"][idx])
        identity = self.df["id"][idx]
        data = data.T if self.trans else data
        if self.data_stats is not None:
            data = (data - self.data_stats[0]) / self.data_stats[-1] 

        feature = {
            "data": self._array_feature(data, dtype = dtype),
            "shape": self._array_feature(data.shape, dtype = tf.int64),
            "id": self._bytes_feature(identity)
        }

        if self.target:
            target = self.df["target"][idx]
            feature["target"] = self._int_feature(target)
        
        example = tf.train.Example(features = tf.train.Features(feature = feature))
        return example.SerializeToString()


    def _serialize_batch(
            self,
            data: Tuple[int, np.ndarray],
            destdir: Path,
            dtype: type = tf.float16,
            filename: str = "train",
            ext_out: str = ".tfrec"
        ) -> None:

        n_batch, batch = data
        filename_batch = destdir.joinpath(filename + str(n_batch).zfill(3) 
                                          + "-" + str(batch.shape[0]) + ext_out)
        with tf.io.TFRecordWriter(str(filename_batch)) as writer:
            for idx in batch:
                writer.write(self._serialize_example(idx, dtype = dtype))


    def serialize_dataset(
            self,
            n_samples,
            destdir: Path,
            dtype: type = tf.float16,
            filename: str = "train",
            ext_out: str = ".tfrec",
            n_processes: int = 1
        ) -> None:

        n_cpus = np.maximum(n_processes, 0)
        n_cpus = np.minimum(n_cpus, mp.cpu_count())

        destdir.mkdir(parents = True, exist_ok = True)

        n_files = np.int32(np.ceil(self.df.shape[0] / n_samples))

        with mp.Pool(n_cpus) as pool:
            writer = partial(self._serialize_batch, destdir = destdir, 
                         dtype = dtype, filename = filename, ext_out = ext_out)
            pool.map(writer, enumerate(np.array_split(self.df.index, n_files)))


    @staticmethod
    def deserialize_example(
            element,
            dtype: type = tf.float16,
            target: bool = False,
            identify: bool = False
        ) -> Tuple[tf.Tensor, int, int]:

        feature = {
            "data"  : tf.io.FixedLenFeature([], tf.string),
            "shape" : tf.io.FixedLenFeature([], tf.string),
            "id" : tf.io.FixedLenFeature([], tf.string)
        }
        if target:
            feature["target"] = tf.io.FixedLenFeature([], tf.int64)

        content  = tf.io.parse_single_example(element, feature)

        shape = tf.io.parse_tensor(content["shape"], out_type = tf.int64)    
        data = tf.io.parse_tensor(content["data"], out_type = dtype)
        data = tf.reshape(data, shape = shape)
        
        label = content["target"] if target else None
        identity = content["id"] if identify else None
        ret_val = (data,)
        ret_val = ret_val + (label,) if target else ret_val
        ret_val = ret_val + (identity,) if identify else ret_val
        return ret_val


##############################################################################