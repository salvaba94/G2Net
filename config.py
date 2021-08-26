# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:38:30 2021

@author: salva
"""

from pathlib import Path
from utilities import GeneralUtilities
import tensorflow as tf


### General data #############################################################
DATA_PATH = Path(".", "..", "raw_data")
TRAIN_PATH = DATA_PATH.joinpath("train")
TEST_PATH = DATA_PATH.joinpath("test")

N_SAMPLES, N_DETECT = GeneralUtilities.get_dims(TRAIN_PATH, trans = True)


### Plotting #################################################################
PLOT_EXAMPLE = False
PLOT_EXAMPLE_IDX = 0

### Data preprocessing #######################################################
N_PROCESSES = 8
DTYPE = tf.float32
tf.keras.backend.set_floatx("float32")


### Training #################################################################
MODEL_TRAIN = True
MODEL_SAVE_NAME = "EffNetB2v2_wintukey_CQTglobalnorm_128_Adam_epoch8_otf.h5"
MODEL_PRELOAD = False
MODEL_PRELOAD_NAME = "EffNetB2v2_wintukey_CQTglobalnorm_128_Adam_epoch8_otf.h5"
MODEL_ID = "B2v2"
MODEL_PATH = Path(".", "..", "saved_models")
AUTOML_PATH = str(Path(".", "automl").absolute())
EFFNETV2_PATH = str(Path(AUTOML_PATH).joinpath("efficientnetv2").absolute())


WINDOW_PRE = ("tukey", 0.25)
SAMPLE_RATE = 2048
F_MIN, F_MAX = 20, 500
HOP_LENGTH = 64
BINS_PER_OCTAVE = 12
WINDOW_CQT = "hann"
TRAINABLE_CQT = False

IMAGE_SHAPE = (128, 128)

SPLIT = 0.9
SEED_SPLIT = 42
BATCH_SIZE = 128
EPOCHS = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4



### Prediction ################################################################
MODEL_PREDICT = False
PREDICTIONS_NAME = "submission.csv"
