# G2Net - Gravitational Wave Detection

<!-- TOC -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
       <ul>
        <li><a href="#contents">Contents</a></li>
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
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
<!-- /TOC -->

## About the Project

Not only a network of Gravitational Waves, Geophysics and Machine Learning experts, G2Net was also released as a [Kaggle Competition](https://www.kaggle.com/c/g2net-gravitational-wave-detection). 
G2Net origin dates back to the discovery of Gravitational Waves (GW) in 2015 ([The Sound of Two Black Holes Colliding](https://www.youtube.com/watch?v=QyDcTbR-kEA)). 
The aim of this competition was to detect GW signals from the mergers of binary black holes. Specifically, the participant was expected to create and train a 
model to analyse synthetic GW time-series data from a network of Earth-based detectors (LIGO Hanford, LIGO Livingston and Virgo). The implementations in this repository 
skyrocketed the ranking to top 8% under certain settings, not meaning with the above that it cannot be further improved.


### Contents

![G2Net Model](https://github.com/salvaba94/G2Net/blob/main/img/Model.png?raw=true "G2Net Model")

### Dependencies
Among others, the project has been built around the following major Python libraries (check ```config/g2net.yml``` for a full list of dependencies):

* [![](https://img.shields.io/badge/Tools-TensorFlow-informational?style=plastic&logo=tensorflow&logoColor=white&color=0e76a8)](https://www.tensorflow.org/)
* [![](https://img.shields.io/badge/Tools-NumPy-informational?style=plastic&logo=numpy&logoColor=white&color=0e76a8)](https://www.numpy.org/)
* [![](https://img.shields.io/badge/Tools-Pandas-informational?style=plastic&logo=pandas&logoColor=white&color=0e76a8)](https://pandas.pydata.org/)
* [![](https://img.shields.io/badge/Tools-SciPy-informational?style=plastic&logo=scipy&logoColor=white&color=0e76a8)](https://www.scipy.org/)


## Getting Started
### Locally
#### Installation
In order to make use of the project locally, one should just follow two steps:
1. Clone the project:
```
  git clone https://github.com/salvaba94/G2Net.git
```
2. Assuming that Anaconda Prompt is installed, run the following command to install the dependencies:
```
  conda env create --file g2net.yml
```

#### Coding

To experiment locally:
1. First, you'll need to manually download the [Competition Data](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data) as the code is not 
going to do it for you to avoid problems with connectivity while downloading a heavy dataset. Paste the content into the ```raw_data``` folder.
2. The controls of the code are in ```src/config.py```. Make sure that, the first time you run the code, any of ```GENERATE_TFR``` or ```GENERATE_NPY``` 
flags are set to ```True```. This will generate standardised datasets in TensorFlow records or NumPy files, respectively.
3. Set to ```False``` these flags and make sure that you are reading the data in the format you generated with the flag ```FROM_TFR```.
3. You are ready to play with the rest of options!


#### Troubleshooting

If by any chance you experience a ```NotImplementedError``` (see below), it is an incompatibility issue between the installed TensorFlow and NumPy library versions. 
It is related to a change in exception types that makes it to be uncaught.

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

### In Colab
Alternatively, feel free to follow the ad-hoc guides in Colab:
* [![](https://img.shields.io/badge/Colab-TPU_Guide-informational?style=plastic&logo=googlecolab&logoColor=white&color=0e76a8)](https://githubtocolab.com/salvaba94/G2Net/blob/main/src/g2net_tpu_colab_full.ipynb) (full version)
* [![](https://img.shields.io/badge/Colab-TPU_Guide-informational?style=plastic&logo=googlecolab&logoColor=white&color=0e76a8)](https://githubtocolab.com/salvaba94/G2Net/blob/main/src/g2net_tpu_colab_short.ipynb) (short version)
* [![](https://img.shields.io/badge/Colab-GPU_Guide-informational?style=plastic&logo=googlecolab&logoColor=white&color=0e76a8)](https://githubtocolab.com/salvaba94/G2Net/blob/main/src/g2net_gpu_colab_short.ipynb)


<!-- ACKNOWL -->
## Acknowledgements

* [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
* [nnAudio: An on-the-fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks](https://arxiv.org/abs/1912.12055)
* [Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic](https://www.aaai.org/Library/ICML/2003/icml03-110.php)
<!-- /ACKNOWL -->


<!-- LINKS -->
[colab-shield]: https://colab.research.google.com/assets/colab-badge.svg
[colab-tpu-full]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[colab-tpu-short]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
<!-- /LINKS -->
