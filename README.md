# Dimensional Causality


## Table of Contents

1. Introduction
2. Installation
   1. C++
   2. Python
   3. R
   4. MatLab
3. Examples
4. TODO


## 1 - Introduction

Let's assume that we have two systems, X and Y. There can be 5 cases of causality:
- X causes Y (direct causality, denoted by X -> Y)
- X and Y both have a causal effect on each other (circular causality, denoted by X <-> Y)
- Y causes X (direct causality, denoted by X <- Y)
- Both X and Y are caused by a third, hidden system (common cause, denoted by X cc Y)
- X and Y are independent (denoted by X | Y)

Very roughly speaking, if X causes Y, then if X changes, Y changes accordingly, but not vice versa. This is fundamentally different from correlation, which is an undirected measure of (linear) dependence. The Dimensional Causality method returns the probability of each of the 5 cases of causality (in the order presented above), given two time-series measured from the two systems. **Keep in mind that Dimensional Causality works only with deterministic, stationary systems** - if you have observational noise, then you should filter it. If your systems have dynamical noise, then you should rather try different methods, like Granger causality.

**Your data probably has to undergo some preprocessing before Dimensional Causality can be effectively applied to it. There are also a few parameters that you have to specify. For a guide on these you should read the paper cited below, especially the Workflow chapter in the Supplementary Material.**

This project contains the implementation of the Dimensional Causality method proposed in **Benko, Zlatniczki, Fabo, Solyom, Eross, Telcs & Somogyvari (2018) - Inference of causal relations via dimensions**.

The method is available in C++, Python, R and MatLab. It is quite fast due to being implemented in pure C++ with a lot of optimization and parallelization. The Python, R and MatLab versions are equally fast, since they rely on the same C++ code.


## 2 - Installation

### 2.1 - C++  
#### 2.1.1 - Prerequisites

- Windows
  - Install mingw
  - Add its bin directory to your system path
- Unix
  - Run
    ```
    apt-get install g++
    apt-get install make
    ```

#### 2.1.2 - Installation
- move to **C++/OpenMP**
- On Windows, run `mingw32-make`
- On Unix, run `make`
- the built dll/so can be found in the C++/OpenMP/bin directory

### 2.2 - Python
#### 2.2.1 - Prerequisites
- Windows
  - Install mingw
  - Add its bin directory to your system path
- Unix
  - Run `apt-get install g++`

#### 2.2.2 - Installation
- move to the **Python** directory
- run `pip install .`

### 2.3 - R
#### 2.3.1 - Prerequisites
- Windows:
  - Install [rtools](https://cran.r-project.org/bin/windows/Rtools/)
  - Make sure that Rtools\bin and Rtools\mingw_32\bin are added to your system path (you should set this during the Rtools install with a checkbox)
- Unix:
  - Run
    ```
    apt-get install g++
    apt-get install make
    ```

#### 2.3.2 - Installation
- move to the **R** directory
- run `R CMD INSTALL dimensionalcausality`


## 3 - Examples

### 3.1 - C++

### 3.2 - Python
```python
import numpy as np
import dimensional_causality as dc

np.random.seed(0)
x = np.random.rand(10000)
y = np.random.rand(10000)
k_range = range(10, 40, 2)

probs = dc.infer_causality(x, y, 4, 1, k_range)
print probs
```

### 3.3 - R
```R
library(dimensionalcausality)

set.seed(0)
x <- runif(10000)
y <- runif(10000)
k_range <- seq(10, 40, 2)

probs <- infer_causality(x, y, 4, 1, k_range)
print(probs)
```

### 3.4 - MatLab


## 4 - TODO

The following list contains future directives.