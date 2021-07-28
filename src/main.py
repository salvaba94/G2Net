# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:33:16 2021

@author: salva
"""

import pandas as pd
from pathlib import Path
from preprocess import PlottingUtilities, GeneralUtilities, LogMelSpectrogram


N_PROCESSES_PRE = 4
RAW_DATA_PATH = Path(".", "..", "raw_data")
RAW_TRAIN_PATH = RAW_DATA_PATH.joinpath("train")
RAW_TEST_PATH = RAW_DATA_PATH.joinpath("test")

MEL_DATA_PATH = Path(".", "..", "mel_data")
MEL_TRAIN_PATH = MEL_DATA_PATH.joinpath("train")
MEL_TEST_PATH = MEL_DATA_PATH.joinpath("test")

SPLIT = 0.9
BATCH_SIZE = 32
SAMPLE_RATE = 2048
FFT_WIN_SIZE = 256
IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128
N_SAMPLES, N_DETECT = GeneralUtilities.get_dims(RAW_TRAIN_PATH, trans = True)
HOP_LENGTH = N_SAMPLES // IMAGE_WIDTH


if __name__ == "__main__":

    sub_file = RAW_DATA_PATH.joinpath("sample_submission.csv")
    train_labels_file = RAW_DATA_PATH.joinpath("training_labels.csv")

    train_df = pd.read_csv(train_labels_file)
    test_df = pd.read_csv(sub_file)

    log_mel = lambda df, path: LogMelSpectrogram(df, path, 
                           sample_rate = SAMPLE_RATE, 
                           n_mels = IMAGE_HEIGHT, 
                           n_fft = FFT_WIN_SIZE - 1, 
                           hop_length = HOP_LENGTH)

    sm_train = log_mel(train_df, RAW_TRAIN_PATH)
    sm_test = log_mel(test_df, RAW_TEST_PATH)
    
    x = GeneralUtilities.get_sample(train_df, RAW_TRAIN_PATH, idx = 0, 
                                    trans = True, target = False)
    y = sm_train.compute_spectrogram(idx = 0)

    print(y.shape)
    PlottingUtilities.plot_wave(x)
    PlottingUtilities.plot_spectrogram(y, SAMPLE_RATE)
    PlottingUtilities.plot_count(train_df)
    
    # sm_train._LogMelSpectrogram__generate_spectrogram(0, MEL_TRAIN_PATH)
    # sm_train.generate_dataset(N_PROCESSES_PRE, MEL_TRAIN_PATH)
    # sm_test.generate_dataset(N_PROCESSES_PRE, MEL_TEST_PATH)



# sample_df = train_df.sample(frac = 1).reset_index(drop = True)
# split = np.floor(sample_df.shape[0] * SPLIT).astype("int32")
# train_df = sample_df[:split]
# val_df = sample_df[split:]

# train_gen = DataGenerator(train_df, TRAIN_PATH, batch_size = BATCH_SIZE)
# val_gen = DataGenerator(val_df, TRAIN_PATH, batch_size = BATCH_SIZE, 
#                         shuffle = False)

# model = LogMelEfficientNetB2(N_SAMPLES, N_DETECT, sample_rate = SAMPLE_RATE, 
#                               fft_size = 256, hop_size = 32, n_mels = 128)
# model.compile(optimizer = tf.keras.optimizers.Adam(), 
#               loss = "binary_crossentropy", 
#               metrics = [tf.keras.metrics.AUC()])
# model.summary()

# train_history = model.fit(
#     train_gen, 
#     epochs = 10,
#     validation_data = val_gen)