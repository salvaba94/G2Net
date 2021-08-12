# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:38:30 2021

@author: salva
"""

from pathlib import Path
from preprocess import GeneralUtilities


### General data #############################################################
RAW_DATA_PATH = Path(".", "..", "raw_data")
RAW_TRAIN_PATH = RAW_DATA_PATH.joinpath("train")
RAW_TEST_PATH = RAW_DATA_PATH.joinpath("test")

N_SAMPLES, N_DETECT = GeneralUtilities.get_dims(RAW_TRAIN_PATH, trans = True)


### Plotting #################################################################
PLOT_EXAMPLE = None
PLOT_EXAMPLE_IDX = 0

### Data preprocessing #######################################################
GENERATE_DATASET = False
USE_CQT = True
N_PROCESSES_PRE = 8
DATA_PATH = Path(".", "..", "cqt_data")
TRAIN_PATH = DATA_PATH.joinpath("train")
TEST_PATH = DATA_PATH.joinpath("test")

SAMPLE_RATE = 2048
F_MIN, F_MAX = 20, 500
HOP_LENGTH = 64
N_MELS = 128
FFT_WIN_SIZE = 256


### Training #################################################################
MODEL_TRAIN = True
MODEL_SAVE_NAME = "EffNetB2v2_CQTglobalnorm_AdamW_CosineLR_128_epoch5.h5"
MODEL_PRELOAD = False
MODEL_PRELOAD_NAME = "EffNetB2v2_CQTglobalnorm_AdamW_CosineLR_128_epoch1.h5"
MODEL_ID = "B2v2"
MODEL_PATH = Path(".", "..", "saved_models")
AUTOML_PATH = str(Path(".", "automl").absolute())
EFFNETV2_PATH = str(Path(AUTOML_PATH).joinpath("efficientnetv2").absolute())

SPLIT = 0.9
SEED_SPLIT = 42
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128

### Prediction ################################################################
MODEL_PREDICT = True
PREDICTIONS_NAME = "submission.csv"
