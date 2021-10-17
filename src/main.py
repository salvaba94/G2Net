# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:33:16 2021

@author: salva
"""

# Import configuration modules
import sys
from config import Config
from pathlib import Path
from git import Repo


# Configure path to use automl efficientnet models
if Config.AUTOML_PATH not in sys.path:
    sys.path.append(Config.AUTOML_PATH)
    sys.path.append(Config.EFFNETV2_PATH)

# Clone AutoML repository if it does not exist locally
if not Path(Config.AUTOML_PATH).is_dir():
    Repo.clone_from(Config.AUTOML_GIT_URL, Config.AUTOML_PATH)


# Import data handling and ML modules
import pandas as pd
import numpy as np
import tensorflow as tf
#from adabelief_tf import AdaBeliefOptimizer

# Import project modules
from utilities import PlottingUtilities, GeneralUtilities
from preprocess import CQTLayer, TukeyWinLayer, BandpassLayer
from preprocess import PermuteChannel, GaussianNoise, SpectralMask
from ingest import TFRDatasetCreator, NPYDatasetCreator, DatasetGeneratorTF
from models import G2NetEfficientNet
from train import RocLoss, Acceleration, CosineAnnealingRestarts



if __name__ == "__main__":

##############################################################################


    # Prepare original dataframes    
    sub_file = Config.RAW_DATA_PATH.joinpath("sample_submission.csv")
    train_labels_file = Config.RAW_DATA_PATH.joinpath("training_labels.csv")

    train_df_ori = pd.read_csv(train_labels_file)
    test_df_ori = pd.read_csv(sub_file)

    # Get raw data mean and std
    wave_stats = GeneralUtilities.get_stats(train_df_ori, Config.RAW_TRAIN_PATH, 
                                            n_processes = Config.N_PROCESSES)


##############################################################################

    # Create strategy and define data types for data and tensorflow models
    strategy, device = Acceleration.get_acceleration()
    dtype = tf.float32


##############################################################################

    # Create preprocessed tensorflow records dataset if requested
    if Config.GENERATE_TFR:
        train_df_random = train_df_ori.sample(
            frac = 1, random_state = Config.SEED_SPLIT).reset_index(drop = True)
        tfr_train = TFRDatasetCreator(train_df_random, Config.RAW_TRAIN_PATH, 
                                        data_stats = wave_stats, trans = True, 
                                        raw_dir = True)
        tfr_test = TFRDatasetCreator(test_df_ori, Config.RAW_TEST_PATH,
                                     data_stats = wave_stats, trans = True,
                                     raw_dir = True)
        tfr_train.serialize_dataset(Config.FILES_PER_TFR, Config.TFR_TRAIN_PATH, 
                                    filename = "train", dtype = dtype)
        tfr_test.serialize_dataset(Config.FILES_PER_TFR, Config.TFR_TEST_PATH, 
                                   filename = "test", dtype = dtype)


    # Create preprocessed numpy dataset if requested
    if Config.GENERATE_NPY:
        npy_train = NPYDatasetCreator(train_df_ori, Config.RAW_TRAIN_PATH, 
                                      data_stats = wave_stats, trans = True, 
                                      raw_dir = True, target = True)
        npy_test = NPYDatasetCreator(test_df_ori, Config.RAW_TEST_PATH,
                                     data_stats = wave_stats, trans = True,
                                     raw_dir = True)
        npy_train.create_dataset(Config.TRAIN_PATH, dtype = np.float32, 
                                 n_processes = Config.N_PROCESSES)
        npy_test.create_dataset(Config.TEST_PATH, dtype = np.float32, 
                                n_processes = Config.N_PROCESSES)


##############################################################################


    # Create datasets from preprocessed tensorflow records splitting labelled dataset 
    # into training and validation set
    if Config.FROM_TFR:
        train_df = pd.DataFrame([x.stem for x in Config.TFR_TRAIN_PATH.glob("*.tfrec")], 
                    columns = ["id"]).sample(frac = 1, random_state = 
                    Config.SEED_SPLIT).reset_index(drop = True)
        test_df = pd.DataFrame([x.stem for x in Config.TFR_TEST_PATH.glob("*.tfrec")], 
                                      columns = ["id"])
        
        n_split = np.int32(train_df.shape[0] * Config.SPLIT)
        training_df = train_df.loc[:n_split - 1, :]
        validation_df = train_df.loc[n_split:, :]
        
        training_gen = DatasetGeneratorTF(training_df, Config.TFR_TRAIN_PATH, 
                                          batch_size = Config.BATCH_SIZE, 
                                          dtype = dtype)
        validation_gen = DatasetGeneratorTF(validation_df, Config.TFR_TRAIN_PATH, 
                                            batch_size = Config.BATCH_SIZE, 
                                            dtype = dtype)
        test_gen = DatasetGeneratorTF(test_df, Config.TFR_TEST_PATH, 
                                      batch_size = Config.BATCH_SIZE_TEST, 
                                      dtype = dtype)
    
        training_ds = training_gen.get_dataset()
        validation_ds = validation_gen.get_dataset()
        test_ds = test_gen.get_dataset(shuffle = False, repeat = False, 
                                       target = False)

        # Estimate number of steps per train, validation and test sets
        ns_training = np.int32(train_df_ori.shape[0] * Config.SPLIT)
        ns_validation = train_df_ori.shape[0] - ns_training
        ns_test = test_df_ori.shape[0]
        spe_training = np.int32(np.ceil(ns_training / Config.BATCH_SIZE))
        spe_validation = np.int32(np.ceil(ns_validation / Config.BATCH_SIZE))
        spe_test = np.int32(np.ceil(ns_test / Config.BATCH_SIZE_TEST))
        
        # Get test IDs from generator
        test_ds_id = test_ds.map(lambda data, identity: tf.strings.unicode_encode(
            identity, "UTF-8"))
        test_ds_id = test_ds_id.unbatch()
        test_ids = next(iter(test_ds_id.batch(test_df_ori.shape[0]))).numpy().astype("U")
        test_ds = test_ds.map(lambda data, identity: data)
        
        data_path = Config.TFR_DATA_PATH

    # Create datasets from preprocessed numpy files splitting labelled dataset 
    # into training and validation set
    else:
        train_df = train_df_ori.sample(
                frac = 1, random_state = Config.SEED_SPLIT).reset_index(drop = True)
        test_df = test_df_ori

        n_split = np.int32(train_df.shape[0] * Config.SPLIT)
        training_df = train_df.loc[:n_split - 1, :]
        validation_df = train_df.loc[n_split:, :]

        training_gen = DatasetGeneratorTF(training_df, Config.TRAIN_PATH, 
                                          batch_size = Config.BATCH_SIZE, 
                                          dtype = dtype, ext = ".npy")
        validation_gen = DatasetGeneratorTF(validation_df, Config.TRAIN_PATH, 
                                            batch_size = Config.BATCH_SIZE, 
                                            dtype = dtype, ext = ".npy")
        test_gen = DatasetGeneratorTF(test_df, Config.TEST_PATH, 
                                      batch_size = Config.BATCH_SIZE_TEST, 
                                      dtype = dtype, ext = ".npy")

        training_ds = training_gen.get_dataset(tfrec = False, repeat = False)
        validation_ds = validation_gen.get_dataset(tfrec = False, repeat = False)
        test_ds = test_gen.get_dataset(tfrec = False, shuffle = False, 
                                       repeat = False, target = False)
        
        spe_training = None
        spe_validation = None
        spe_test = None
        
        test_ids = test_df["id"]
        
        data_path = Config.DATA_PATH


##############################################################################


    # Create model, compile and display summary within the scope of the 
    # distribution strategy
    tf.keras.backend.clear_session()
    with strategy.scope():
        model_gen = G2NetEfficientNet(input_shape = 
                                      (Config.N_SAMPLES, Config.N_DETECT),
                                      window_shape = Config.TUKEY_SHAPE,
                                      trainable_window = Config.TRAINABLE_TUKEY,
                                      sample_rate = Config.SAMPLE_RATE,
                                      degree_filt = Config.DEGREE_FILT,
                                      f_band_filt = Config.F_BAND_FILT,
                                      trainable_filt = Config.TRAINABLE_FILT,
                                      hop_length = Config.HOP_LENGTH,
                                      f_band_spec = Config.F_BAND_SPEC,
                                      bins_per_octave = Config.BINS_PER_OCTAVE,
                                      window_cqt = Config.WINDOW_CQT,
                                      resize_shape = 
                                      (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                                      p_perm = Config.P_PERM,
                                      p_mask = Config.P_MASK, 
                                      n_max_mask_t = Config.N_MAX_MASK,
                                      w_mask_t = Config.W_MASK,
                                      n_max_mask_f = Config.N_MAX_MASK,
                                      w_mask_f = Config.W_MASK,
                                      strategy = device)
        model = model_gen.get_model(effnet_id = Config.MODEL_ID)
        optimizer = tf.keras.optimizers.Adam(learning_rate = Config.LEARNING_RATE)
        # lr = CosineAnnealingRestarts(initial_learning_rate = Config.LEARNING_RATE, 
        #                              first_decay_steps = 1000, alpha = 0.01)
        # optimizer = AdaBeliefOptimizer(learning_rate = Config.LEARNING_RATE, 
        #                                amsgrad = False, print_change_log = False)
        model.compile(optimizer = optimizer, 
                  loss = "binary_crossentropy", 
                  # loss = RocLoss(),
                  metrics = [tf.keras.metrics.AUC()])
        model.summary()

##############################################################################

    # Preload model weights
    if Config.MODEL_PRELOAD:        
        pretrained_model = Config.MODEL_PATH.joinpath(Config.MODEL_PRELOAD_NAME)
        if tf.io.gfile.isdir(pretrained_model):
            pretrained_model = tf.train.latest_checkpoint(pretrained_model)
        model.load_weights(pretrained_model)
        
    with strategy.scope():
        model.get_layer("cqt").trainable = Config.TRAINABLE_CQT

##############################################################################


    # Train model with training and validation sets with checkpoints and control 
    # over training validation loss plateaus
    if Config.MODEL_TRAIN:
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience = 1,
            monitor = "val_loss", cooldown = 0, verbose = 1)

        check_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = Config.MODEL_PATH.joinpath("ckpt-{epoch:d}"),
            save_weights_only = True, 
            monitor = "val_auc",
            mode = "max",
            save_best_only = True)

        train_history = model.fit(training_ds, epochs = Config.EPOCHS,
                                  batch_size = Config.BATCH_SIZE,
                                  validation_data = validation_ds,
                                  steps_per_epoch = spe_training,
                                  validation_steps = spe_validation,
                                  callbacks = [lr_callback, check_callback])
        
        model.save_weights(Config.MODEL_PATH.joinpath(Config.MODEL_SAVE_NAME))


##############################################################################

    # Predict on test set and save to submission file
    if Config.MODEL_PREDICT:

        preds_test = model.predict(test_ds, batch_size = Config.BATCH_SIZE_TEST, 
                                   steps = spe_test, verbose = 1)

        sub_df = pd.DataFrame({
            "id": test_ids,
            "target": preds_test.flatten()
        })
        sub_df = sub_df.sort_values("id").reset_index(drop = True)
        
        sub_df.to_csv(data_path.joinpath(Config.PREDICTIONS_NAME), 
                      index = False)

##############################################################################

    # Plot an example if requested
    if Config.PLOT_EXAMPLE:

        permute = model.get_layer("permute")
        window = model.get_layer("window")
        bandpass = model.get_layer("bandpass")
        cqt = model.get_layer("cqt")
        resize = model.get_layer("resize")
        permute = model.get_layer("permute")
        mask = model.get_layer("mask")
        effnet = model.get_layer(Config.MODEL_ID)
        flatten = model.get_layer("flatten")
        dense = model.get_layer("dense")
        
        if Config.PLOT_TEST:
            x = GeneralUtilities.get_sample(test_df_ori, Config.RAW_TEST_PATH, 
                                            idx = Config.PLOT_EXAMPLE_IDX, raw_dir = True,
                                            trans = True, target = False).astype(np.float32)
        else:
            x = GeneralUtilities.get_sample(train_df_ori, Config.RAW_TRAIN_PATH, 
                                            idx = Config.PLOT_EXAMPLE_IDX, raw_dir = True,
                                            trans = True, target = False).astype(np.float32)

        x_sc = (x - wave_stats[0]) / wave_stats[-1]
        x_ref = x_sc[np.newaxis, ...]
        y = x_ref
        
        y = window(y)
        y_win = np.squeeze(y.numpy())
        y = bandpass(y)
        y_band = np.squeeze(y.numpy())
        y = cqt(y, training = False)
        y = resize(y)
        y_spec = np.squeeze(y.numpy())
        y = permute(y, training = True)
        y = mask(y, training = True)
        y_masked = np.squeeze(y.numpy())
        y = effnet(y)
        y = flatten(y)
        y = dense(y)
        y_dense = np.squeeze(y.numpy())
        
        PlottingUtilities.plot_wave(x_sc)
        PlottingUtilities.plot_wave(y_win)
        PlottingUtilities.plot_wave(y_band)
        
        PlottingUtilities.plot_spectrogram(y_spec)
        PlottingUtilities.plot_spectrogram(y_masked)
        PlottingUtilities.plot_count(train_df_ori)

##############################################################################

