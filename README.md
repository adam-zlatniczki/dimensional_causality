# Dimensional Causality


## Table of Contents

1. Terms and conditions
3. Introduction
3. Installation
   1. C++
   3. Python
   3. R
   4. MatLab
4. Examples
5. TODO


## 1 - Terms and conditions of use

This software is licenced under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3 - for the exact details, please read **licence.txt**. If you use the software you must also cite the original paper.

## 2 - Introduction

Let's assume that we have two systems, X and Y. There can be 5 cases of causality:
- X causes Y (direct causality, denoted by X -> Y)
- X and Y both have a causal effect on each other (circular causality, denoted by X <-> Y)
- Y causes X (direct causality, denoted by X <- Y)
- Both X and Y are caused by a third, hidden system (common cause, denoted by X cc Y)
- X and Y are independent (denoted by X | Y)

Very roughly speaking, if X causes Y, then if X changes, Y changes accordingly, but not vice versa. This is fundamentally different from correlation, which is an undirected measure of (linear) dependence. The Dimensional Causality method returns the probability of each of the 5 cases of causality (in the order presented above), given two time-series measured from the two systems. **Keep in mind that Dimensional Causality works only with deterministic, stationary systems** - if you have observational noise, then you should filter it. If your systems have dynamical noise, then you should rather try different methods, like Granger causality.

**Your data probably has to undergo some preprocessing before Dimensional Causality can be effectively applied to it. There are also a few parameters that you have to specify. For a guide on these you should read the paper cited below, especially the Workflow chapter in the Supplementary Material.**

This project contains the implementation of the Dimensional Causality method proposed in **Benko, Zlatniczki, Fabo, Solyom, Eross, Telcs & Somogyvari (2018) - Exact Inference of Causal Relation in Dynamical Systems**.

The method is available in C++, Python, R and MatLab. It is quite fast due to being implemented in pure C++ with a lot of optimization and parallelization. The Python, R and MatLab versions are equally fast, since they rely on the same C++ code.


## 3 - Installation

The install process assumes that your Python/R environment was built on the same architecture as your processor. This means that if you have a 64 bit OS but use 32 bit Python or R, then the package won't work. In that case you manually have to modify the install scripts by adding the '-m32' flag to the g++ commands.
Typical installation time depends on your machine, but in general, a few minutes should be sufficient.
The installation process was tested with GNU g++ 7.2.0, PIP 18.0, Rtools 3.4.0.1964.

### 3.1 - C++  
#### 3.1.1 - Prerequisites

- Windows
  - Install mingw
  - Add its bin directory to your system path
- Unix
  - Run
    ```
    apt-get install g++
    apt-get install make
    ```

#### 3.1.2 - Installation
- move to **C++/OpenMP**
- On Windows, run `mingw32-make`
- On Unix, run `make`
- the built dll/so can be found in the C++/OpenMP/bin directory

### 3.2 - Python
#### 3.3.1 - Prerequisites
- Windows
  - Install mingw
  - Add its bin directory to your system path
- Unix
  - Run `apt-get install g++`

#### 3.3.2 - Installation
- move to the **root** directory, where you can find setup.py
- run `pip install .`

### 3.3 - R
#### 3.3.1 - Prerequisites
- Windows:
  - Install [rtools](https://cran.r-project.org/bin/windows/Rtools/)
  - Make sure that Rtools\bin and Rtools\mingw_32\bin are added to your system path (you should set this during the Rtools install with a checkbox)
- Unix:
  - Run
    ```
    apt-get install g++
    apt-get install make
    ```

#### 3.3.2 - Installation
- move to the **R** directory
- run `R CMD INSTALL dimensionalcausality`


## 4 - Examples

In this example we generate two random, independent uniform time-series and check their causal relation. We expect the final probabilities to be [0.026, 0.069, 0.016, 0.118, 0.771] (very slight differences are acceptable, since the time-permuted manifold Z is constructed randomly). Running the demo should take a couple of seconds only.

### 4.1 - C++

### 4.2 - Python
```python
import numpy as np
import dimensional_causality as dc

np.random.seed(0)
x = np.random.rand(10000)
y = np.random.rand(10000)
k_range = range(10, 40, 2)

probs, dims, stdevs = dc.infer_causality(x, y, 4, 1, k_range)
print probs
```

### 4.3 - R
```R
library(dimensionalcausality)

set.seed(0)
x <- runif(10000)
y <- runif(10000)
k_range <- seq(10, 40, 2)

probs <- infer_causality(x, y, 4, 1, k_range)
print(probs)
```

### 4.4 - MatLab


## 5 - TODO

The following list contains future directives.