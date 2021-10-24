# G2Net - Gravitational Wave Detection

<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stars][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL License][license-shield]][license-url]

<!-- TOC -->
<details open=true>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
       <ul>
        <li><a href="#contents">Contents</a></li>
        <ul>
          <li><a href="#the-model">The Model</a></li>
          <li><a href="#major-files">Major Files</a></li>
        </ul>
      </ul>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#locally">Locally</a></li>
        <ul>
          <li><a href="#installation">Installation</a></li>
          <li><a href="#coding">Coding</a></li>
          <li><a href="#troubleshooting">Troubleshooting</a></li>
        </ul>
        <li><a href="#in-colab">In Colab</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgements">Acknowledgments</a></li>
  </ol>
</details>
<!-- /TOC -->


<!-- ABOUT -->
## About the Project

Not only a network of Gravitational Waves, Geophysics and Machine Learning experts, G2Net was also released as a [Kaggle Competition][kaggle-competition]. G2Net origin dates back to the discovery of Gravitational Waves (GW) in 2015 ([The Sound of Two Black Holes Colliding][colliding-blackholes]). The aim of this competition was to detect GW signals from the mergers of binary black holes. Specifically, the participant was expected to create and train a model to analyse synthetic GW time-series data from a network of Earth-based detectors (LIGO Hanford, LIGO Livingston and Virgo). The implementations in this repository skyrocketed the ranking (AUC score on test set) to top 8% under certain settings, not meaning with the above that it cannot be further improved.


### Contents

#### The Model
The model implemented for the competition (see the image below) has been created following an end-to-end philosophy, meaning that even the time-series pre-processing logic is included as part of the model and might be made trainable. To know more details about the building blocks of the model, refer to any of the [Colab Guides](#in-colab) provided by the project.

![G2Net Model][model-image]

<p align="right"><a href="#top">Back to top</a></p>

#### Major Files
The major project source code files are listed below in a tree-like fashion:

```bash
    G2Net
      └───src
          │   config.py
          │   main.py
          ├───ingest
          │       DatasetGeneratorTF.py
          │       NPYDatasetCreator.py
          │       TFRDatasetCreator.py
          ├───models
          │       ImageBasedModels.py
          ├───preprocess
          │       Augmentation.py
          │       Preprocessing.py
          │       Spectrogram.py
          ├───train
          │       Acceleration.py
          │       Losses.py
          │       Schedulers.py
          └───utilities
                  GeneralUtilities.py
                  PlottingUtilities.py
```


The most important elements in the project are outlined and described as follows:
* ```config.py```: Contains a configuration class with the parameters used by the model or the training process and other data ingestion options.
* ```main.py```: Implements the functionality to train and predict with the model locally in GPU/CPU.
* Ingest module:
  * ```NPYDatasetCreator.py```: Implements the logic to standardise the full dataset on a multiprocessing fashion.
  * ```TFRDatasetCreator.py```: Implements the logic to standardise, encode, create and decode TensorFlow records. 
  * ```DatasetGeneratorTF.py```: Includes a class implementing functionality to create TensorFlow Datasets pipelines from both TensorFlow records and NumPy files.
* Models module:
  * ```ImageBasedModels.py```: Includes a Keras model based on 2D convolutions preceded by a pre-processing phase culminated with the generation of a spectrogram or similar. The 2D convolutional model is here an [EfficientNet v2][efficientnet].
* Preprocess module:
  * ```Augmentation.py```: Implements several augmentations in the form of Keras layers, including Gaussian noise, spectral masking (TPU-compatible and TPU-incompatible versions) and channel permutation.
  * ```Preprocessing.py```: Implements several preprocessing layers in the form of trainable Keras layers, including time windows (TPU-incompatible Tukey window and generic TPU-compatible window), bandpass filtering and spectral whitening.
  * ```Spectrogram.py```: Includes a TensorFlow version of CQT1992v2 implemented in [nnAudio][nnaudio] with PyTorch. Being in the form of a Keras layer, it also adds functionality to adapt the output range to that recommended as per stability by 2D convolutional models.
* Train module:
  * ```Acceleration.py```: Includes the logic to automatically configure the TPU if any.
  * ```Losses.py```: Implements a differentiable loss whose minimisation directly maximises the AUC score.
  * ```Schedulers.py```: Implements a wrapper to make CosineDecayRestarts learning rate scheduler compatible with ReduceLROnPlateau.
* Utilities module:
  * ```GeneralUtilities.py```: General utilities used all along the project mainly to perform automatic Tensor broadcast and determine mean and standard deviation from a dataset with multiprocessing capabilities.
  * ```PlottingUtilities.py```: Includes all the logic behind the plots.   

<p align="right"><a href="#top">Back to top</a></p>

### Dependencies
Among others, the project has been built around the following major Python libraries (check ```config/g2net.yml``` for a full list of dependencies with tested versions):

* [![][tensorflow-logo]][tensorflow-link] (version 2.x)
* [![][numpy-logo]][numpy-link]
* [![][pandas-logo]][pandas-link]
* [![][scipy-logo]][scipy-link]

<p align="right"><a href="#top">Back to top</a></p>
<!-- /ABOUT -->


<!-- START -->
## Getting Started
### Locally
#### Installation
In order to make use of the project locally (tested in Windows), one should just follow two steps:
1. Clone the project:
```
  git clone https://github.com/salvaba94/G2Net.git
```
2. Assuming that Anaconda Prompt is installed, run the following command to install the dependencies:
```
  conda env create --file g2net.yml
```

<p align="right"><a href="#top">Back to top</a></p>

#### Coding

To experiment locally:
1. First, you'll need to manually download the [Competition Data][kaggle-data] as the code is not going to do it for you to avoid problems with connectivity (while downloading a heavy dataset). Paste the content into the ```raw_data``` folder.
2. The controls of the code are in ```src/config.py```. Make sure that, the first time you run the code, any of ```GENERATE_TFR``` or ```GENERATE_NPY``` flags are set to ```True```. This will generate standardised datasets in TensorFlow records or NumPy files, respectively.
3. Set to ```False``` these flags and make sure that you are reading the data in the format you generated with the flag ```FROM_TFR```.
4. You are ready to play with the rest of options!

<p align="right"><a href="#top">Back to top</a></p>

#### Troubleshooting

If by any chance you experience a ```NotImplementedError``` (see below), it is an incompatibility issue between the installed TensorFlow and NumPy library versions. It is related to a change in exception types that makes it to be uncaught.

```
  NotImplementedError: Cannot convert a symbolic Tensor (gradient_tape/model/bandpass/irfft_2/add:0) to a numpy array. 
  This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported.
``` 

The origin is in line 867 in ```tensorflow/python/framework/ops.py```. It is solved by replacing

```
  def __array__(self):
    raise NotImplementedError(
        "Cannot convert a symbolic Tensor ({}) to a numpy array."
        " This error may indicate that you're trying to pass a Tensor to"
        " a NumPy call, which is not supported".format(self.name))
```
by
```
  def __array__(self):
    raise TypeError(
        "Cannot convert a symbolic Tensor ({}) to a numpy array."
        " This error may indicate that you're trying to pass a Tensor to"
        " a NumPy call, which is not supported".format(self.name))
```

<p align="right"><a href="#top">Back to top</a></p>

### In Colab
Alternatively, feel free to follow the ad-hoc guides in Colab:
* [![][colab-tpu-logo]][colab-tpu-guide-f] (full version)
* [![][colab-tpu-logo]][colab-tpu-guide-s] (short version)
* [![][colab-gpu-logo]][colab-gpu-guide]

**Important note**: As the notebooks connect with your Google Drive to save trained models, copy them to your Drive and run them from there not from the link. Anyway, Google is going to notify you that the notebooks have been loaded from GitHub and not from your Drive.

<p align="right"><a href="#top">Back to top</a></p>
<!-- /START -->

<!-- CONTRIBUTING -->
## Contributing

Any contributions are greatly appreciated. If you have suggestions that would make the project any better, fork the repository and create a pull request or simply open an issue. If you decide to follow the first procedure, here is a reminder of the steps:

1. Fork the project.
2. Create your branch:
```
  git checkout -b branchname
```
3. Commit your changes:
```
  git commit -m "Add some amazing feature"
```
4. Push to the branch: 
```
  git push origin branchname
```
5. Open a pull request.

<p align="right"><a href="#top">Back to top</a></p>

<!-- ACKNOWL -->
## Acknowledgements

* [EfficientNetV2: Smaller Models and Faster Training][efficientnet]
* [nnAudio: An on-the-fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks][nnaudio]
* [Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic][rocloss]
* [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients][adabelief]
* [Darien Schettler][darien-notebook] (for solving with his amazing notebooks issues I had while using EfficientNet v2 with pretrained weights)

<p align="right"><a href="#top">Back to top</a></p>
<!-- /ACKNOWL -->

**If you like the project and/or any of this contents results useful to you, don't forget to give it a star! It means a lot to me 😄**


<!-- LINKS -->
[contributors-shield]: https://img.shields.io/github/contributors/salvaba94/G2Net.svg?style=plastic&color=0e76a8
[contributors-url]: https://github.com/salvaba94/G2Net/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/salvaba94/G2Net.svg?style=plastic&color=0e76a8
[forks-url]: https://github.com/salvaba94/G2Net/network/members
[stars-shield]: https://img.shields.io/github/stars/salvaba94/G2Net.svg?style=plastic&color=0e76a8
[stars-url]: https://github.com/salvaba94/G2Net/stargazers
[issues-shield]: https://img.shields.io/github/issues/salvaba94/G2Net.svg?style=plastic&color=0e76a8
[issues-url]: https://github.com/salvaba94/G2Net/issues
[license-shield]: https://img.shields.io/github/license/salvaba94/G2Net.svg?style=plastic&color=0e76a8
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[kaggle-competition]: https://www.kaggle.com/c/g2net-gravitational-wave-detection
[colliding-blackholes]: https://www.youtube.com/watch?v=QyDcTbR-kEA
[model-image]: https://github.com/salvaba94/G2Net/blob/main/img/Model.png?raw=true
[tensorflow-logo]: https://img.shields.io/badge/Tools-TensorFlow-informational?style=plastic&logo=tensorflow&logoColor=white&color=0e76a8
[tensorflow-link]: https://www.tensorflow.org/
[numpy-logo]: https://img.shields.io/badge/Tools-NumPy-informational?style=plastic&logo=numpy&logoColor=white&color=0e76a8
[numpy-link]: https://www.numpy.org/
[pandas-logo]: https://img.shields.io/badge/Tools-Pandas-informational?style=plastic&logo=pandas&logoColor=white&color=0e76a8
[pandas-link]: https://pandas.pydata.org/
[scipy-logo]: https://img.shields.io/badge/Tools-SciPy-informational?style=plastic&logo=scipy&logoColor=white&color=0e76a8
[scipy-link]: https://www.scipy.org/
[colab-tpu-logo]: https://img.shields.io/badge/Colab-TPU_Guide-informational?style=plastic&logo=googlecolab&logoColor=white&color=0e76a8
[colab-gpu-logo]: https://img.shields.io/badge/Colab-GPU_Guide-informational?style=plastic&logo=googlecolab&logoColor=white&color=0e76a8
[colab-tpu-guide-f]:  https://githubtocolab.com/salvaba94/G2Net/blob/main/src/g2net_tpu_colab_full.ipynb
[colab-tpu-guide-s]: https://githubtocolab.com/salvaba94/G2Net/blob/main/src/g2net_tpu_colab_short.ipynb
[colab-gpu-guide]: https://githubtocolab.com/salvaba94/G2Net/blob/main/src/g2net_gpu_colab_short.ipynb
[kaggle-data]: https://www.kaggle.com/c/g2net-gravitational-wave-detection/data
[efficientnet]: https://arxiv.org/abs/2104.00298
[nnaudio]: https://arxiv.org/abs/1912.12055
[rocloss]: https://www.aaai.org/Library/ICML/2003/icml03-110.php
[adabelief]: https://arxiv.org/abs/2010.07468
[darien-notebook]: https://www.kaggle.com/dschettler8845/load-efficientnetv2-pretrained-weights

<!-- /LINKS -->
