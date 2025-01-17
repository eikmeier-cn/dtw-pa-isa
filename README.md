[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14679872.svg)](https://doi.org/10.5281/zenodo.14679872)

# Dynamic Time Warping Proximity Analysis (DTW-PA) and Intermediate Signal Approach (DTW-ISA)

Welcome to the dtw-pa-isa code repository for the Dynamic Time Warping Proximity Analysis (DTW-PA) and the Dynamic Time Warping Intermediate Signal Approach (DTW-ISA). These methods are designed to assess and mitigate cycle-skipping in Full-Waveform Inversion (FWI).

## Methods Overview

### Dynamic Time Warping Proximity Analysis (DTW-PA)

DTW-PA provides a qualitative assessment of the suitability of the initial or current model to mitigate cycle-skipping in FWI. This analysis evaluates whether the model is well-positioned to achieve reliable inversion results. The method is detailed in the paper:
**Full-waveform inversion cycle-skipping mitigation with dynamic time warping, Part 1: a method for a proximity analysis between the current and the observed** (reference to be provided soon).

### Dynamic Time Warping Intermediate Signal Approach (DTW-ISA)

DTW-ISA mitigates cycle-skipping by generating intermediate signals between the modelled and observed (true) signals. These signals are positioned close enough to the modelled signals to serve as temporary targets, allowing the inversion to progress incrementally and guiding the solution gradually toward the observed data. The method is detailed in the paper:
**Full-waveform inversion cycle-skipping mitigation with dynamic time warping, Part 2: an intermediate signal approach** (reference to be provided soon).

## Code

This repository provides an implementation of DTW-PA and DTW-ISA designed for ease of use and straightforward integration into your existing FWI framework.

## Instalation

Clone the repository into the folder of your choice.

    $ git clone https://github.com/eikmeier-cn/dtw-pa-isa.git

Access the dtw-pa-isa folder. To manage dependencies more effectively, consider using a [Python virtual environment](https://docs.python.org/3/library/venv.html). If so, create and activate your virtual environment and install the dependencies with the `requirements.txt` file.

    $ python -m pip install --upgrade pip
    $ pip install -r requirements.txt

Additionally, you may want to install Jupyter Notebook.

    $ pip install notebook

## Tutorials

[Here](https://github.com/eikmeier-cn/dtw-pa-isa/tree/main/dtw-pa-isa) you will find two tutorials on Jupyter Notebook that will help you understand how to use the code.

## Dependencies

- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)
- [SciPy](https://www.scipy.org)
- [DTAIDistance](https://github.com/wannesm/dtaidistance)
- [Matplotlib](https://matplotlib.org)
- [Cycler](https://matplotlib.org/cycler/)

## Citing

If you publish results using dtw-pa-isa code, DTW-PA or DTW-ISA, we would be grateful if you would cite the following papers.

- [Full-waveform inversion cycle-skipping mitigation with dynamic time warping, Part 1: a method for a proximity analysis between the current and the observed]() (reference to be provided soon).
- [Full-waveform inversion cycle-skipping mitigation with dynamic time warping, Part 2: an intermediate signal approach]() (reference to be provided soon).
- [PhD Thesis]() (reference to be provided soon).

## Get in touch

Claus Naves Eikmeier [Linkedin](https://www.linkedin.com/in/claus-naves-eikmeier/)