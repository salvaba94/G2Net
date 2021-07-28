# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:11:12 2021

@author: salva
"""

import tensorflow as tf
import numpy as np
from pathlib import Path


class DataGeneratorKeras(tf.keras.utils.Sequence):
    
    def __init__(self, dataframe, datadir, batch_size = 32, 
                  shuffle = True, target = True, ext = ".npy"):
        self.datadir = Path(datadir)
        self.df = dataframe
        self.shuffle = shuffle
        self.target = target
        self.batch_size = batch_size
        self.ext = ext
        self.on_epoch_end()

    
    def __len__(self):
        return np.floor(self.df.shape[0] / self.batch_size).astype("int32")
    
    def __getitem__(self, idx):
        
        def _append(signals, datadir, fname):
            signal_path = datadir.joinpath(fname[0], fname[1], fname[2], fname)
            signal = np.load(signal_path).T [np.newaxis, ...]
            signals_out = np.hstack((signals, signal)) if signals.size else signal
            return signals_out

        start_idx = idx * self.batch_size
        batch = self.df[start_idx : start_idx + self.batch_size]
        
        signals = np.array([]).astype("float32")

        for fname in batch.id:
            signals = _append(signals, self.datadir, fname + self.ext)

        return_val = (signals, batch.target.values) if self.target else signals
        return return_val

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac = 1).reset_index(drop = True)