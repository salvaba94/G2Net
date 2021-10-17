# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:38:30 2021

@author: salva
"""

from pathlib import Path


class Config:
### General data #############################################################
    RAW_DATA_PATH = Path(".", "..", "raw_data")
    RAW_TRAIN_PATH = RAW_DATA_PATH.joinpath("train")
    RAW_TEST_PATH = RAW_DATA_PATH.joinpath("test")

    N_PROCESSES = 8

    N_SAMPLES, N_DETECT = 4096, 3

### Data preprocessing #######################################################
    GENERATE_TFR = False
    FILES_PER_TFR = 11200
    TFR_DATA_PATH = Path(".", "..", "tfr_data_float32")
    TFR_TRAIN_PATH = TFR_DATA_PATH.joinpath("train")
    TFR_TEST_PATH = TFR_DATA_PATH.joinpath("test")

    GENERATE_NPY = False
    DATA_PATH = Path(".", "..", "data")
    TRAIN_PATH = DATA_PATH.joinpath("train")
    TEST_PATH = DATA_PATH.joinpath("test")

### Training #################################################################
    FROM_TFR = False
    MODEL_TRAIN = True
    MODEL_SAVE_NAME = "Model_publication_ref_small.h5"
    MODEL_PRELOAD = False
    MODEL_PRELOAD_NAME = "Model_publication_ref_small.h5"
    MODEL_PATH = Path(".", "..", "saved_models")
    AUTOML_PATH = str(Path(".", "automl").absolute())
    EFFNETV2_PATH = str(Path(AUTOML_PATH).joinpath("efficientnetv2").absolute())
    AUTOML_GIT_URL = "https://github.com/google/automl.git"

    SPLIT = 0.98
    SEED_SPLIT = 21
    BATCH_SIZE = 128
    BATCH_SIZE_TEST = 32
    EPOCHS = 4
    LEARNING_RATE = 0.0001
    
### Prediction ################################################################
    MODEL_PREDICT = False
    PREDICTIONS_NAME = "submission.csv"


### Model ####################################################################
    TUKEY_SHAPE = 0.2
    TRAINABLE_TUKEY = True

    DEGREE_FILT = 6
    F_BAND_FILT = (20., 500.)
    TRAINABLE_FILT = True

    SAMPLE_RATE = 2048
    F_BAND_SPEC = (20., 500.)
    HOP_LENGTH = 64
    BINS_PER_OCTAVE = 12
    WINDOW_CQT = "hann"
    TRAINABLE_CQT = True
    
    IMAGE_SIZE = 65

    P_PERM = 1.

    P_MASK = 1.
    N_MAX_MASK = 2
    W_MASK = (0, IMAGE_SIZE // 6)
    
    MODEL_ID = "efficientnetv2-b0"

### Plotting #################################################################
    PLOT_EXAMPLE = False
    PLOT_TEST = False
    PLOT_EXAMPLE_IDX = 98 #44438 in test can be perfectly seen