# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:33:16 2021

@author: salva
"""

# Import configuration modules
import sys
import os
import config

# Configure path to use automl efficientnet models
if config.AUTOML_PATH not in sys.path:
    sys.path.append(config.AUTOML_PATH)
    sys.path.append(config.EFFNETV2_PATH)

# Import data handling and ML modules
import pandas as pd
import numpy as np
import tensorflow as tf
from adabelief_tf import AdaBeliefOptimizer

# Import project modules
from utilities import PlottingUtilities, GeneralUtilities
from preprocess import CQTLayer, TukeyWinLayer, WhitenLayer, BandpassLayer
from ingest import TFRDatasetCreator, NPYDatasetCreator, DatasetGeneratorTF
from models import G2NetEfficientNet
from train import RocLoss, Acceleration, CosineAnnealingRestarts



if __name__ == "__main__":

##############################################################################
    

    # Prepare original dataframes    
    sub_file = config.RAW_DATA_PATH.joinpath("sample_submission.csv")
    train_labels_file = config.RAW_DATA_PATH.joinpath("training_labels.csv")

    train_df_ori = pd.read_csv(train_labels_file)
    test_df_ori = pd.read_csv(sub_file)

    # Get raw data mean and std
    wave_stats = GeneralUtilities.get_stats(train_df_ori, config.RAW_TRAIN_PATH, 
                                            n_processes = config.N_PROCESSES)


##############################################################################

    # Plot an example if requested
    if config.PLOT_EXAMPLE:
        window = TukeyWinLayer(initial_alpha = config.TUKEY_SHAPE)

        bandpass = BandpassLayer(sample_rate = config.SAMPLE_RATE, 
                                 f_band = (config.F_MIN_FILT, config.F_MAX_FILT), 
                                 n_samples = config.N_SAMPLES)
        cqt = CQTLayer(sample_rate = config.SAMPLE_RATE, 
                       hop_length = config.HOP_LENGTH, 
                       f_band = (config.F_MIN_SPEC, config.F_MAX_SPEC), 
                       trainable = False)
        
        x = GeneralUtilities.get_sample(train_df_ori, config.RAW_TRAIN_PATH, 
                                        idx = config.PLOT_EXAMPLE_IDX, 
                                        trans = True, target = False).astype(np.float32)
        x_sc = (x - wave_stats[0])/wave_stats[-1]
        x = x_sc[np.newaxis, ...]

        y = window(x)
        y_win = np.squeeze(y.numpy())
        # y = whiten(y)
        # y_whi = np.squeeze(y.numpy())
        y = bandpass(y)
        y_band = np.squeeze(y)
        y = cqt(y, training = True)
        y_spec = np.squeeze(y.numpy())

        PlottingUtilities.plot_wave(x_sc)
        PlottingUtilities.plot_wave(y_win)
        # PlottingUtilities.plot_wave(y_whi)
        PlottingUtilities.plot_wave(y_band)
        PlottingUtilities.plot_spectrogram(y_spec, config.SAMPLE_RATE)
        PlottingUtilities.plot_count(train_df_ori)


##############################################################################

    # Create strategy and define data types for data and tensorflow models
    strategy, device = Acceleration.get_acceleration()
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_bfloat16") if \
        device == "TPU" else tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    dtype = tf.bfloat16 if device == "TPU" else tf.float16
    tf.keras.mixed_precision.experimental.set_policy(policy)
    dtype_map = {
        tf.float16: np.float16,
        tf.bfloat16: np.float16,
        tf.float32: np.float32,
        tf.float64: np.float64,
    }


##############################################################################

    # Create preprocessed tensorflow records dataset if requested
    if config.GENERATE_TFR:
        train_df_random = train_df_ori.sample(
            frac = 1, random_state = config.SEED_SPLIT).reset_index(drop = True)
        tfr_train = TFRDatasetCreator(train_df_random, config.RAW_TRAIN_PATH, 
                                        data_stats = wave_stats, trans = True, 
                                        raw_dir = True, target = True)
        tfr_test = TFRDatasetCreator(test_df_ori, config.RAW_TEST_PATH,
                                     data_stats = wave_stats, trans = True,
                                     raw_dir = True)
        tfr_train.serialize_dataset(config.FILES_PER_TFR, config.TFR_TRAIN_PATH, 
                                    filename = "train", dtype = dtype, 
                                    n_processes = config.N_PROCESSES)
        tfr_test.serialize_dataset(config.FILES_PER_TFR, config.TFR_TEST_PATH, 
                                   filename = "test", dtype = dtype, 
                                   n_processes = config.N_PROCESSES)


    # Create preprocessed numpy dataset if requested
    if config.GENERATE_NPY:
        npy_train = NPYDatasetCreator(train_df_ori, config.RAW_TRAIN_PATH, 
                                      data_stats = wave_stats, trans = True, 
                                      raw_dir = True, target = True)
        npy_test = NPYDatasetCreator(test_df_ori, config.RAW_TEST_PATH,
                                     data_stats = wave_stats, trans = True,
                                     raw_dir = True)
        npy_train.create_dataset(config.TRAIN_PATH, dtype = dtype_map[dtype], 
                                 n_processes = config.N_PROCESSES)
        npy_test.create_dataset(config.TEST_PATH, dtype = dtype_map[dtype], 
                                n_processes = config.N_PROCESSES)


##############################################################################


    # Create datasets from preprocessed tensorflow records splitting labelled dataset 
    # into training and validation set
    if config.FROM_TFR:
        train_df = pd.DataFrame([x.stem for x in config.TRAIN_PATH.glob("*.tfrec")], 
                    columns = ["id"]).sample(frac = 1, random_state = 
                    config.SEED_SPLIT).reset_index(drop = True)
        test_df = pd.DataFrame([x.stem for x in config.TEST_PATH.glob("*.tfrec")], 
                                      columns = ["id"])
        
        n_split = np.int32(train_df.shape[0] * config.SPLIT)
        training_df = train_df.loc[:n_split, :]
        validation_df = train_df.loc[n_split:, :]
        
        training_gen = DatasetGeneratorTF(training_df, config.TFR_TRAIN_PATH, 
                                          batch_size = config.BATCH_SIZE, 
                                          dtype = dtype)
        validation_gen = DatasetGeneratorTF(validation_df, config.TFR_TRAIN_PATH, 
                                            batch_size = config.BATCH_SIZE, 
                                            dtype = dtype)
        test_gen = DatasetGeneratorTF(test_df, config.TFR_TEST_PATH, 
                                      batch_size = config.BATCH_SIZE, 
                                          dtype = dtype)
    
        training_ds = training_gen.get_dataset()
        validation_ds = validation_gen.get_dataset()
        test_ds = test_gen.get_dataset(shuffle = False, ignore_order = False, 
                                       repeat = False, target = False, 
                                       identify = True)

        # Estimate number of steps per train, validation and test sets
        ns_training = np.int32(train_df_ori.shape[0] * config.SPLIT)
        ns_validation = train_df_ori.shape[0] - ns_training
        ns_test = test_df_ori.shape[0]
        spe_training = np.int32(np.ceil(ns_training / config.BATCH_SIZE))
        spe_validation = np.int32(np.ceil(ns_validation / config.BATCH_SIZE))
        spe_test = np.int32(np.ceil(ns_test / config.BATCH_SIZE))
        
        # Get test IDs from generator
        test_ids = [identity.numpy().decode("UTF-8") for _, identity in \
                    iter(test_ds.unbatch())]

    # Create datasets from preprocessed numpy files splitting labelled dataset 
    # into training and validation set
    else:
        train_df = train_df_ori.sample(
                frac = 1, random_state = config.SEED_SPLIT).reset_index(drop = True)
        test_df = test_df_ori

        n_split = np.int32(train_df.shape[0] * config.SPLIT)
        training_df = train_df.loc[:n_split, :]
        validation_df = train_df.loc[n_split:, :]

        training_gen = DatasetGeneratorTF(training_df, config.TRAIN_PATH, 
                                          batch_size = config.BATCH_SIZE, 
                                          dtype = dtype, ext = ".npy")
        validation_gen = DatasetGeneratorTF(validation_df, config.TRAIN_PATH, 
                                            batch_size = config.BATCH_SIZE, 
                                            dtype = dtype, ext = ".npy")
        test_gen = DatasetGeneratorTF(test_df, config.TEST_PATH, 
                                      batch_size = config.BATCH_SIZE, 
                                      dtype = dtype, ext = ".npy")

        training_ds = training_gen.get_dataset(tfrec = False, repeat = False)
        validation_ds = validation_gen.get_dataset(tfrec = False, repeat = False)
        test_ds = test_gen.get_dataset(tfrec = False, shuffle = False, 
                                       repeat = False, target = False)
        
        spe_training = None
        spe_validation = None
        spe_test = None
        
        test_ids = test_df["id"]


##############################################################################


    # Create model, compile and display summary within the scope of the 
    # distribution strategy
    with strategy.scope():
        model_gen = G2NetEfficientNet(input_shape = 
                                      (config.N_SAMPLES, config.N_DETECT),
                                      window_shape = config.TUKEY_SHAPE,
                                      trainable_window = config.TRAINABLE_TUKEY,
                                      sample_rate = config.SAMPLE_RATE,
                                      degree_filt = config.FILTER_DEGREE,
                                      f_band_filt = (config.F_MIN_FILT, config.F_MAX_FILT),
                                      trainable_filt = config.TRAINABLE_FILTER,
                                      hop_length = config.HOP_LENGTH,
                                      f_band_spec = (config.F_MIN_SPEC, config.F_MAX_SPEC),
                                      bins_per_octave = config.BINS_PER_OCTAVE,
                                      window_cqt = config.WINDOW_CQT, 
                                      trainable_cqt = config.TRAINABLE_CQT,
                                      resize_shape = config.IMAGE_SHAPE)
        model = model_gen.get_model(effnet_id = config.MODEL_ID)
        optimizer = tf.keras.optimizers.Adam(learning_rate = config.LEARNING_RATE)
        # lr = CosineAnnealingRestarts(initial_learning_rate = config.LEARNING_RATE, 
        #                              first_decay_steps = 1000, alpha = 0.01)
        # wd = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate = config.WEIGHT_DECAY, decay_steps = 3500,
        #     decay_rate = 0.96)
        # optimizer = AdaBeliefOptimizer(learning_rate = lr, weight_decay = wd, 
        #                                amsgrad = True, rectify = True,
        #                                print_change_log = False)
        model.compile(optimizer = optimizer, 
                  loss = "binary_crossentropy", 
                  metrics = [tf.keras.metrics.AUC()])
        model.summary()

##############################################################################

    # Preload model weights
    if config.MODEL_PRELOAD:
        model.load_weights(config.MODEL_PATH.joinpath(config.MODEL_PRELOAD_NAME))

##############################################################################


    # Train model with training and validation sets with checkpoints and control 
    # over training validation loss plateaus
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
                                  steps_per_epoch = spe_training,
                                  validation_steps = spe_validation,
                                  callbacks = [lr_callback, check_callback])
        
        model.save_weights(config.MODEL_PATH.joinpath(config.MODEL_SAVE_NAME))


##############################################################################


    # Predict on test set and save to submission file
    if config.MODEL_PREDICT:

        preds_test = model.predict(test_ds, steps = spe_test, verbose = 1)

        sub_df = pd.DataFrame({
            "id": test_ids,
            "target": preds_test.flatten()
        })
        sub_df = sub_df.sort_values("id").reset_index(drop = True)
        
        sub_df.to_csv(config.DATA_PATH.joinpath(config.PREDICTIONS_NAME), 
                      index = False)


##############################################################################

