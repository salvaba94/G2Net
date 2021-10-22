# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 11:32:58 2021

@author: salva
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os
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


    @staticmethod
    def _bytes_feature(
            value: Union[np.ndarray, tf.Tensor, str]
        ) -> tf.train.Feature:
        """
        Function to convert an array, string or tensor to a list of bytes feature

        Parameters
        ----------
        value : Union[np.ndarray, tf.Tensor, str]
            Input value to convert to a list of bytes feature.

        Returns
        -------
        tf.train.Feature
            The converted feature.
        """

        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))


    @staticmethod
    def _int_feature(
            value: Union[int, bool]
        ) -> tf.train.Feature:
        """
        Function to convert a bool / enum / int / uint to an int feature.
        
        Parameters
        ----------
        value : Union[int, bool]
            Input value to convert to an int feature.
        
        Returns
        -------
        tf.train.Feature
            The converted feature.
        """

        return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))


    def _serialize_example(
            self, 
            idx: int,
            dtype: type = tf.float32
        ) -> str:
        """
        Method to serialise a single example.
        
        Parameters
        ----------
        idx : int
            ID of the example to selialize.
        dtype : type, optional
            Data type to which the example should be serialised. The default 
            is tf.float32.

        Returns
        -------
        str
            Serialised example.
        """

        data = np.load(self.df["path"][idx])
        identity = self.df["id"][idx]
        target = self.df["target"][idx]

        data = data.T if self.trans else data
        
        if self.data_stats is not None:
            data = (data - self.data_stats[0]) / self.data_stats[-1] 

        data = tf.convert_to_tensor(data, dtype = dtype)

        feature = {
            "data": self._bytes_feature(tf.io.serialize_tensor(data)),
            "id": self._bytes_feature(identity.encode()),
            "target": self._int_feature(np.int(target))
        }
        
        example = tf.train.Example(features = tf.train.Features(feature = feature))
        return example.SerializeToString()


    def _serialize_batch(
            self,
            data: Tuple[int, np.ndarray],
            destdir: Path,
            dtype: type = tf.float32,
            filename: str = "train",
            ext_out: str = ".tfrec"
        ) -> None:
        """
        Method to serialise a batch of examples and write it to tensorflow record.
        
        Parameters
        ----------
        data : Tuple[int, np.ndarray]
            Batch data. Contains batch ID and array of examples to be serialised,
            respectively.
        destdir : Path
            Destination directory.
        dtype : type, optional
            Data type to which the examples should be serialised. The default 
            is tf.float32.
        filename : str, optional
            Filename of the output tensorflow record. To it, the batch ID and 
            the range of the examples contained in it will be appended. 
            The default is "train".
        ext_out : str, optional
            Extension of the output files. The default is ".tfrec".
        """

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
            dtype: type = tf.float32,
            filename: str = "train",
            ext_out: str = ".tfrec"
        ) -> None:
        """
        Method to serialise a the full dataset and write it to a set of 
        tensorflow records.
        
        Parameters
        ----------
        n_samples : int
            Number of samples used to calculate the number of tensorflow 
            records files.
        destdir : Path
            Destination directory.
        dtype : type, optional
            Data type to which the examples should be serialised. The default 
            is tf.float32.
        filename : str, optional
            Filename of the output tensorflow record. The default is "train".
        ext_out : str, optional
            Extension of the output files. The default is ".tfrec".
        """

        destdir.mkdir(parents = True, exist_ok = True)

        n_files = np.int32(np.ceil(self.df.shape[0] / n_samples))

        for n_batch, batch in enumerate(np.array_split(self.df.index, n_files)): 
            print("Writing TFRecord " + str(n_batch) + " with files from " + 
                  str(batch[0]) + " to " + str(batch[-1]))
            writer = partial(self._serialize_batch, destdir = destdir, 
                         dtype = dtype, filename = filename, ext_out = ext_out)
            writer((n_batch, batch))


    @staticmethod
    def deserialize_example(
            element,
            dtype: type = tf.float32,
            target: bool = False,
            shape: Tuple[int, int] = (4096, 3)
        ) -> Tuple[tf.Tensor, int]:
        """
        Method intended to be used in any dataset generator to deserialise the 
        examples from tensorflow records.
        
        Parameters
        ----------
        element : int
            Serialised example.
        dtype : type, optional
            Data type to which the example was serialised. The default is tf.float32.
        target : bool, optional
            Whether the label should be included in the output. If false, it is 
            interpreted that the aim is to predict and, therefore, the ID of the 
            example is appended instead. The default is False.
        shape : Tuple[int, int], optional
            Shape to which the example data should be reshaped 
        
        Returns
        -------
        Tuple[tf.Tensor, int]
            Example data and label or ID, depending on the value of target argument.
        """

        feature = {
            "data"  : tf.io.FixedLenFeature([], tf.string),
            "id" : tf.io.FixedLenFeature([], tf.string),
            "target" : tf.io.FixedLenFeature([], tf.int64)
        }

        content = tf.io.parse_single_example(element, feature)
 
        data = content["data"]
        data = tf.io.parse_tensor(data, out_type = dtype)
        data = tf.reshape(data, shape = shape)

        idx = tf.cast(tf.strings.unicode_decode(content["id"], "UTF-8"), 
                      dtype = tf.int32)
        label = tf.cast(content["target"] , dtype)
        
        if target:
            return data, label
        else:
            return data, idx


##############################################################################