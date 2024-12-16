# Digit recognition using Logistic Regression, Softmax and Gradient Descent
> This repository contains Python scripts for implementing softmax and logistic regression models to recognize handwritten digits from the MNIST dataset.
> 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Navigation

- [Overview](#overview)
- [Features](#Features)
- [Results](#Results)
- [Conclusion](#Conclusion)

## Overview

### Goals

- To implement Softmax function to distinguish between the numbers in the dataset.
- To implement gradient descent and Newton methods to minimize functions.
- To try different stepwise methods(Powel Wolfe, Armijo, constant) and understand their pros and cons.

### Dataset 

- MNIST dataset

### Model

- **Softmax Regression**: A multi-class logistic regression model using the softmax function.
- **Logistic Regression**: A binary logistic regression model for digit classification.
- **Gradient Descent**: An optimization algorithm used to minimize a function iteratively by adjusting its parameters.
- **Armijo method**: A line search method in optimization that ensures sufficient decrease of the objective function by adjusting the step size iteratively.
- **Powel Wolfe method**: A line search method in optimization that combines the Armijo and the Wolfe conditions for ensuring sufficient decrease and curvature.

## Features

- **Implementation**: Both softmax and logistic regression models were implemented in Python without relying on external libraries except for numpy.
- **Performance**: Achieved accuracy scores of 92% and 99% on the test set for softmax and logistic regression models, respectively.
- **Optimization**: Explored and compared different step size functions for gradient descent optimization.

## Results

- **Logistic Regression with constant learning rate**:
  
![constant_lreg]
- **Softmax Regression with constant learning rate**:
  
![constant_softmax]
- **Logistic Regression with Armijo step size**:
  
![armijo_lreg]
- **Softmax Regression with Armijo step size**:
  
![armijo_softmax]
- **Logistic Regression with Powell-Wolfe step size**:
  
![powell_wolfe_lreg]
- **Softmax Regression with Powell-Wolfe step size**:
  
![powell_wolfe_softmax]

## Conclusion

This project implemented and optimized machine learning models for digit recognition using softmax and logistic regression techniques in Python. Achieving high accuracies of 92% and 99% on test datasets for softmax and logistic regression, respectively, underscored their effectiveness in classifying MNIST handwritten digits. Optimization methods like gradient descent with Armijo and Powell-Wolfe rules were explored for performance enhancement. However, these methods proved less effective for softmax regression due to slower convergence compared to a constant learning rate. The Newton method was ineffective for both models due to a singular Hessian matrix.


[constant_lreg]: results/constant_logistic_regression.png
[constant_softmax]: results/constant_softmax.png
[armijo_lreg]: results/armijo_logistic_regression.png
[armijo_softmax]: results/armijo_softmax.png
[powell_wolfe_lreg]: results/powell_wolfe_logistic_regression.png
[powell_wolfe_softmax]: results/powell_wolfe_softmax.png
