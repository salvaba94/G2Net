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

from preprocess import PlottingUtilities, GeneralUtilities
from preprocess import LogMelSpectrogram, CQTransform
from ingest import DatasetGeneratorTF
from models import G2NetEfficientNet



if __name__ == "__main__":

    sub_file = config.RAW_DATA_PATH.joinpath("sample_submission.csv")
    train_labels_file = config.RAW_DATA_PATH.joinpath("training_labels.csv")

    train_df = pd.read_csv(train_labels_file)
    test_df = pd.read_csv(sub_file)


    if config.PLOT_EXAMPLE or config.GENERATE_DATASET:
        
        if config.USE_CQT:
            preprocessor = lambda df, path: CQTransform(df, path,
                           sample_rate = config.SAMPLE_RATE, 
                           hop_length = config.HOP_LENGTH,
                           f_min = config.F_MIN,
                           f_max = config.F_MAX)
        else:
            preprocessor = lambda df, path: LogMelSpectrogram(df, path,
                           sample_rate = config.SAMPLE_RATE, 
                           n_mels = config.N_MELS, 
                           n_fft = config.FFT_WIN_SIZE, 
                           hop_length = config.HOP_LENGTH,
                           f_min = config.F_MIN,
                           f_max = config.F_MAX)
    
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
        sm_train.generate_dataset(config.N_PROCESSES_PRE, config.TRAIN_PATH, 
                                  dtype = np.float16)
        sm_test.generate_dataset(config.N_PROCESSES_PRE, config.TEST_PATH,
                                  dtype = np.float16)

    n_split = np.int32(train_df.shape[0] * config.SPLIT)
    train_df = train_df.sample(frac = 1, random_state = config.SEED_SPLIT
                                ).reset_index(drop = True)

    training_df = train_df.loc[:n_split, :]
    validation_df = train_df.loc[n_split:, :]
        
    training_gen = DatasetGeneratorTF(training_df, config.TRAIN_PATH, 
                                      batch_size = config.BATCH_SIZE,
                                      image_size = [config.IMAGE_WIDTH, 
                                                    config.IMAGE_HEIGHT])
    validation_gen = DatasetGeneratorTF(validation_df, config.TRAIN_PATH, 
                                        batch_size = config.BATCH_SIZE,
                                        image_size = [config.IMAGE_WIDTH, 
                                                      config.IMAGE_HEIGHT])
    test_gen = DatasetGeneratorTF(test_df, config.TEST_PATH, 
                                  batch_size = config.BATCH_SIZE,
                                  image_size = [config.IMAGE_WIDTH, 
                                                config.IMAGE_HEIGHT],
                                  shuffle = False, target = False)
        
    training_ds = training_gen.get_dataset()
    validation_ds = validation_gen.get_dataset()
    test_ds = test_gen.get_dataset()


    model_gen = G2NetEfficientNet(input_shape = (config.IMAGE_WIDTH, 
                                                 config.IMAGE_HEIGHT, 
                                                 config.N_DETECT))
    model = model_gen.get_model(effnet_id = config.MODEL_ID)
    model.compile(optimizer = tf.keras.optimizers.Adam(lr = config.LEARNING_RATE), 
                  loss = "binary_crossentropy", 
                  metrics = [tf.keras.metrics.AUC()])
    model.summary()
        
    if config.MODEL_PRELOAD:
        model.load_weights(config.MODEL_PATH.joinpath(config.MODEL_PRELOAD_NAME))
        
    if config.MODEL_TRAIN:
        train_history = model.fit(training_ds, epochs = config.EPOCHS,
                                  validation_data = validation_ds)
        
        model.save_weights(config.MODEL_PATH.joinpath(config.MODEL_SAVE_NAME))
            
    if config.MODEL_PREDICT:
        preds = model.predict(test_ds, verbose = 1)
        
        sub_df = pd.DataFrame({
            "id":test_df["id"],
            "target": preds.ravel()
        })
        
        sub_df.to_csv(config.DATA_PATH.joinpath(config.PREDICTIONS_NAME), 
                      index = False)



