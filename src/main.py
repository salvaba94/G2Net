# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:33:16 2021

@author: salva
"""

import sys
import config

if config.AUTOML_PATH not in sys.path:
    sys.path.append(config.AUTOML_PATH)
    sys.path.append(config.EFFNETV2_PATH)

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from automl.efficientnetv2 import utils

from preprocess import PlottingUtilities, GeneralUtilities
from preprocess import LogMelSpectrogram, CQTransform
from ingest import DatasetGeneratorTF
from models import G2NetEfficientNet
from train import RocLoss
from train import Acceleration


def main():
    sub_file = config.RAW_DATA_PATH.joinpath("sample_submission.csv")
    train_labels_file = config.RAW_DATA_PATH.joinpath("training_labels.csv")

    train_df = pd.read_csv(train_labels_file)
    test_df = pd.read_csv(sub_file)
    
    # y_true = train_df["target"].to_numpy().astype(np.float32)[:, np.newaxis]
    
    # noise = np.random.normal(0., 0.1, y_true.shape).astype(np.float32)
    
    # y_pred = np.clip(y_true + noise, 0., 1.)
    # lossTorch = RocStarLossTorch()
    # lossTF = RocStarLoss()

    # loss_val_torch = lossTorch.forward(torch.from_numpy(y_pred[:config.BATCH_SIZE, :]), 
    #                     torch.from_numpy(y_true[:config.BATCH_SIZE, :]))
    # print("loss torch: ",  loss_val_torch)

    # loss_val_TF = lossTF.call(tf.convert_to_tensor(y_true[:config.BATCH_SIZE, :]),
    #                           tf.convert_to_tensor(y_pred[:config.BATCH_SIZE, :]))
    

    # print("loss TF: ", loss_val_TF)


    if config.PLOT_EXAMPLE or config.GENERATE_DATASET:
        
        if config.USE_CQT:
            preprocessor = lambda df, path: CQTransform(df, path,
                            n_processes = config.N_PROCESSES_PRE,
                            sample_rate = config.SAMPLE_RATE, 
                            hop_length = config.HOP_LENGTH,
                            f_band = (config.F_MIN, config.F_MAX),
                            sample_size = train_df.shape[0])
        else:
            preprocessor = lambda df, path: LogMelSpectrogram(df, path,
                            n_processes = config.N_PROCESSES_PRE,
                            sample_rate = config.SAMPLE_RATE, 
                            n_mels = config.N_MELS, 
                            n_fft = config.FFT_WIN_SIZE, 
                            hop_length = config.HOP_LENGTH,
                            f_band = (config.F_MIN, config.F_MAX))
    
        sm_train = preprocessor(train_df, config.RAW_TRAIN_PATH)
        sm_test = preprocessor(test_df, config.RAW_TEST_PATH)
    
    
    if config.PLOT_EXAMPLE:
        x = GeneralUtilities.get_sample(train_df, config.RAW_TRAIN_PATH, 
                                        idx = config.PLOT_EXAMPLE_IDX, 
                                        trans = True, target = False)
        y = sm_train.compute_spectrogram(idx = config.PLOT_EXAMPLE_IDX)

        PlottingUtilities.plot_wave(x)
        PlottingUtilities.plot_spectrogram(y, config.SAMPLE_RATE)
        PlottingUtilities.plot_count(train_df)
    
    if config.GENERATE_DATASET:
        sm_train.generate_dataset(config.TRAIN_PATH, dtype = np.float16)
        sm_test.generate_dataset(config.TEST_PATH, dtype = np.float16)

    n_split = np.int32(train_df.shape[0] * config.SPLIT)
    train_df = train_df.sample(frac = 1, random_state = config.SEED_SPLIT
                                ).reset_index(drop = True)

    training_df = train_df.loc[:n_split, :]
    validation_df = train_df.loc[n_split:, :]

    
    strategy, device = Acceleration.get_acceleration()
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_bfloat16") if \
        device == "TPU" else tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    dtype = tf.bfloat16 if device == "TPU" else tf.float16
    tf.keras.mixed_precision.experimental.set_policy(policy)
        
    training_gen = DatasetGeneratorTF(training_df, config.TRAIN_PATH, 
                                      batch_size = config.BATCH_SIZE, dtype = dtype,
                                      image_size = [config.IMAGE_WIDTH, 
                                                    config.IMAGE_HEIGHT])
    validation_gen = DatasetGeneratorTF(validation_df, config.TRAIN_PATH, 
                                        batch_size = config.BATCH_SIZE, dtype = dtype,
                                        image_size = [config.IMAGE_WIDTH, 
                                                      config.IMAGE_HEIGHT])
    test_gen = DatasetGeneratorTF(test_df, config.TEST_PATH, 
                                  batch_size = config.BATCH_SIZE, dtype = dtype,
                                  image_size = [config.IMAGE_WIDTH, 
                                                config.IMAGE_HEIGHT],
                                  shuffle = False, target = False)
        
    training_ds = training_gen.get_dataset()
    validation_ds = validation_gen.get_dataset()
    test_ds = test_gen.get_dataset()


    model_gen = G2NetEfficientNet(dtype = dtype, 
                                  input_shape = (config.IMAGE_WIDTH, 
                                                 config.IMAGE_HEIGHT, 
                                                 config.N_DETECT))

    optimizer = tf.keras.optimizers.Adam(learning_rate = config.LEARNING_RATE)

    with strategy.scope():
        model = model_gen.get_model(effnet_id = config.MODEL_ID)
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
            filepath = config.MODEL_PATH,
            save_weights_only = True, 
            monitor = "val_auc",
            mode = "max",
            save_best_only = True)
        train_history = model.fit(training_ds, epochs = config.EPOCHS,
                                  validation_data = validation_ds,
                                  callbacks = [lr_callback, check_callback])
        
        model.save_weights(config.MODEL_PATH.joinpath(config.MODEL_SAVE_NAME))
            
    if config.MODEL_PREDICT:
        preds = model.predict(test_ds, verbose = 1)
        
        sub_df = pd.DataFrame({
            "id":test_df["id"],
            "target": preds.ravel()
        })
        
        sub_df.to_csv(config.DATA_PATH.joinpath(config.PREDICTIONS_NAME), 
                      index = False)


if __name__ == "__main__":
    main()


