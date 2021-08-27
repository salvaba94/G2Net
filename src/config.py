# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:38:30 2021

@author: salva
"""

from pathlib import Path
from utilities import GeneralUtilities


### General data #############################################################
RAW_DATA_PATH = Path(".", "..", "raw_data")
RAW_TRAIN_PATH = RAW_DATA_PATH.joinpath("train")
RAW_TEST_PATH = RAW_DATA_PATH.joinpath("test")

N_SAMPLES, N_DETECT = GeneralUtilities.get_dims(RAW_TRAIN_PATH, trans = True)


### Plotting #################################################################
PLOT_EXAMPLE = True
PLOT_EXAMPLE_IDX = 98 #98


### Data preprocessing #######################################################

N_PROCESSES = 8

GENERATE_TFR = False
FILES_PER_TFR = 8000
TFR_DATA_PATH = Path(".", "..", "tfr_data")
TFR_TRAIN_PATH = TFR_DATA_PATH.joinpath("train")
TFR_TEST_PATH = TFR_DATA_PATH.joinpath("test")

GENERATE_NPY = False
DATA_PATH = Path(".", "..", "data")
TRAIN_PATH = DATA_PATH.joinpath("train")
TEST_PATH = DATA_PATH.joinpath("test")



### Training #################################################################
FROM_TFR = False
MODEL_TRAIN = False
MODEL_SAVE_NAME = "EffNetB0v2_globnorm_tukey_CQT_64_Adam_epoch6_otf.h5"
MODEL_PRELOAD = False
MODEL_PRELOAD_NAME = "EffNetB0v2_globnorm_tukey_CQT_64_Adam_epoch6_otf.h5"
MODEL_ID = "B0v2"
MODEL_PATH = Path(".", "..", "saved_models")
AUTOML_PATH = str(Path(".", "automl").absolute())
EFFNETV2_PATH = str(Path(AUTOML_PATH).joinpath("efficientnetv2").absolute())


TUKEY_SHAPE = 0.2
TRAINABLE_TUKEY = True

FILTER_DEGREE = 3
F_MIN_FILT, F_MAX_FILT = 20., 500.
TRAINABLE_FILTER = False

SAMPLE_RATE = 2048
F_MIN_SPEC, F_MAX_SPEC = 20., 500.
HOP_LENGTH = 64
BINS_PER_OCTAVE = 12
WINDOW_CQT = "hann"
TRAINABLE_CQT = False
IMAGE_SHAPE = (64, 64)


SPLIT = 0.9
SEED_SPLIT = 42
BATCH_SIZE = 128
EPOCHS = 6
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3



### Prediction ################################################################
MODEL_PREDICT = False
PREDICTIONS_NAME = "submission.csv"
