# Prime factors :1234::crystal_ball:

:page_facing_up: _Factorization Learning Large Factorizing semiprimes using ML & RNS_ (Dorenbos et al., 2020)  
:pencil2: Haico Dorenbos, Lindsay Kempen, Michelle Peters and Eric van Schaik  
:books: Full text [requestable](mailto:research@linths.com)

## Paper abstract

RSA is one of the most prominent forms of encryption nowadays. Its security depends on the assumption that it is very difficult to determine the prime factorization of a semiprime (the product of two prime numbers). Here an attempt is made to find a relationship between a semiprime and its prime factorization in polynomial time of the bit length of the semiprime used. This is done by generating a number of semiprimes with their smallest prime factor, converting them into a residue number system, performing a sine/cosine translation on them and training a Linear Multivariate Gaussian System (LMGS) with the resulting feature vectors, using a Maximum Likelihood Estimator (MLE). Unfortunately, the resulting distributions approximate a uniform distribution, leaving RSA safe for now.

## Instructions

Python 3.7 is advised. Several libraries are necessary; standard ones (`numpy`, `sklearn`, `matplotlib`, etc.), but most importantly `Crypto` if you want to generate new primes.
Below, you can click the non-abstract files to navigate to them.

### Run it

1. If you do not want to test the model or write the model weights, uncomment the corresponding lines at the very end of [`main.py`](src/main.py).
2. Run [`main.py`](src/main.py) in the `src` folder. It trains the models (and if specified, test the models and write weights).

### Read it

* Result of training  
The models are in a self-defined subfolder in `data`, (defined in `MODEL_FOLDER` in [`main.py`](src/main.py)). One system model is actually a dictionary of LMGS models, to make up a full prediction system.
* Result of testing  
The stats are in a stats folder within the aforementioned self-defined subfolder (defined in `STATS_FOLDER` in [`main.py`](src/main.py)). It contains:
  * `summary.txt`  
  All settings of the run and textual results, such as the average normalized loglikelihood and confusion matrices per residue.
  * `normalized likelihoods (x of y).png`  
  Likelihood plot of residue outcomes for the specific residue class `x`. There is one plot per residue class.
  * `normalized ranks per residue.png`  
  Plots the normalized likelihood rank of the actual residue, per residue class. A datapoint of (5,0.0) means that for residue class 5, the actual residue was predicted as the most likely. (7,0.9) means for residue class 7, the actual residue was just among the least likely 10%.

### Tweak it

To tweak the program, change in [`main.py`](src/main.py) the appropriate global variables that are in the _Settings_ block. These are the following variables. If a model doesn't already exist for the specified settings, it will train one using the pre-generated prime list [`train_primes_#40000.p`](data/train_primes_#40000.p). Note this takes very long. If the pre-generated is not long enough, it will generate new primes. This will add to the training time.
* `BIT_LENGTH`: of the semiprime
* `NO_TRAIN`: number of train samples
* `NO_TEST`: number of test samples
* `WITHOUT_ZERO`: whether the sine/cosine translation ignores the zero point
* `NO_MODS`: number of residue classes used for features
* `MAKE_POLY`: add polynomial features of a certain degree
* `LIM_MODELS`: speed up the process by only predicting the first residues of the prime factor
* `DATA_SUBFOLDER`: specify a folder name to separate data

## Copyright

This project is licensed under the MIT License.

```
Copyright (c) 2020 Haico Dorenbos, Lindsay Kempen, Michelle Peters and Eric van Schaik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
