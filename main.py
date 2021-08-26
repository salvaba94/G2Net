# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:33:16 2021

@author: salva
"""

import sys
import os
import config

if config.AUTOML_PATH not in sys.path:
    sys.path.append(config.AUTOML_PATH)
    sys.path.append(config.EFFNETV2_PATH)

import pandas as pd
import numpy as np
import tensorflow as tf


from utilities import PlottingUtilities, GeneralUtilities
from preprocess import CQTLayer, WindowingLayer
from ingest import DatasetGeneratorTF
from models import G2NetEfficientNet
from train import RocLoss

if "TF_KERAS" not in os.environ:
    os.environ["TF_KERAS"] = "1"
from keras_adamw import AdamW



if __name__ == "__main__":
    sub_file = config.DATA_PATH.joinpath("sample_submission.csv")
    train_labels_file = config.DATA_PATH.joinpath("training_labels.csv")

    train_df = pd.read_csv(train_labels_file)
    test_df = pd.read_csv(sub_file)
    
    wave_stats = GeneralUtilities.get_stats(train_df, config.TRAIN_PATH, 
                                            n_processes = config.N_PROCESSES)

    if config.PLOT_EXAMPLE:
        window = WindowingLayer(window = ("tukey", 0.25), 
                                window_len = config.N_SAMPLES)
        cqt = CQTLayer(sample_rate = config.SAMPLE_RATE, 
                       hop_length = config.HOP_LENGTH, 
                       f_band = (config.F_MIN, config.F_MAX), trainable = False)
        
        x = GeneralUtilities.get_sample(train_df, config.TRAIN_PATH, 
                                        idx = config.PLOT_EXAMPLE_IDX, 
                                        trans = True, target = False).astype(np.float32)

        x = (x - wave_stats[0])/wave_stats[-1]
        x_sc = x[np.newaxis, ...]
        
        y = window(x_sc)
        y_win = np.squeeze(y.numpy())
        y = cqt(y, training = True)
        y_spec = np.swapaxes(np.squeeze(y.numpy()), 0, 1)

        PlottingUtilities.plot_wave(x)
        PlottingUtilities.plot_wave(y_win)
        PlottingUtilities.plot_spectrogram(y_spec, config.SAMPLE_RATE)
        PlottingUtilities.plot_count(train_df)


    n_split = np.int32(train_df.shape[0] * config.SPLIT)
    train_df = train_df.sample(frac = 1, random_state = config.SEED_SPLIT
                                 ).reset_index(drop = True)
    training_df = train_df.loc[:n_split, :]
    validation_df = train_df.loc[n_split:, :]

     
    training_gen = DatasetGeneratorTF(training_df, config.TRAIN_PATH, 
                                      batch_size = config.BATCH_SIZE, 
                                      dtype = config.DTYPE)
    validation_gen = DatasetGeneratorTF(validation_df, config.TRAIN_PATH, 
                                        batch_size = config.BATCH_SIZE, 
                                        dtype = config.DTYPE)
    test_gen = DatasetGeneratorTF(test_df, config.TEST_PATH, 
                                  batch_size = config.BATCH_SIZE, 
                                  dtype = config.DTYPE)
        
    training_ds = training_gen.get_dataset()
    validation_ds = validation_gen.get_dataset()


    model_gen = G2NetEfficientNet(input_shape = (config.N_SAMPLES, config.N_DETECT),
                                  wave_stats = wave_stats,
                                  window_pre = config.WINDOW_PRE,
                                  sample_rate = config.SAMPLE_RATE,
                                  hop_length = config.HOP_LENGTH,
                                  f_band = (config.F_MIN, config.F_MAX),
                                  bins_per_octave = config.BINS_PER_OCTAVE,
                                  window_cqt = config.WINDOW_CQT, 
                                  trainable_cqt = config.TRAINABLE_CQT,
                                  resize_shape = config.IMAGE_SHAPE,
                                  dtype = config.DTYPE)

    model = model_gen.get_model(effnet_id = config.MODEL_ID)
    # lr_multipliers = {"windowing/window": 0.1, "cqt/real_kernels": 0.01, 
    #                   "cqt/imag_kernels": 0.01}
    # optimizer = AdamW(learning_rate = config.LEARNING_RATE, 
    #                   model = model, use_cosine_annealing = True, 
    #                   total_iterations = 1000, 
    #                   lr_multipliers = lr_multipliers)
    optimizer = tf.keras.optimizers.Adam(learning_rate = config.LEARNING_RATE)

    model.compile(optimizer = optimizer, 
                  loss = "binary_crossentropy", 
                  metrics = [tf.keras.metrics.AUC()])
    model.summary()

    
    if config.MODEL_PRELOAD:
        model.load_weights(config.MODEL_PATH.joinpath(config.MODEL_PRELOAD_NAME))
        
    if config.MODEL_TRAIN:
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience = 1,
            monitor = "val_loss", cooldown = 0, verbose = 1)

        check_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = config.MODEL_PATH.joinpath("ckpt-{epoch:d}"),
            save_weights_only = True, 
            monitor = "val_auc",
            mode = "max",
            save_best_only = True)

        train_history = model.fit(training_ds, epochs = config.EPOCHS,
                                  validation_data = validation_ds,
                                  callbacks = [lr_callback, check_callback])
        
        model.save_weights(config.MODEL_PATH.joinpath(config.MODEL_SAVE_NAME))


    if config.MODEL_PREDICT:
        test_ds = test_gen.get_dataset(shuffle = False, repeat = False, 
                                       target = False)
        preds_test = model.predict(test_ds, verbose = 1)
        
        
        sub_df = pd.DataFrame({
            "id": test_df["id"],
            "target": preds_test.ravel()
        })
        
        sub_df.to_csv(config.DATA_PATH.joinpath(config.PREDICTIONS_NAME), 
                      index = False)


