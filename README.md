# LCBNN

This repository contains code used in the experiments of our paper: ["Loss-calibrated approximate inference in Bayesian neural networks"](https://arxiv.org/abs/1805.03901) by Adam D. Cobb, Stephen J. Roberts and Yarin Gal.

## Abstract 
Current approaches in approximate inference for Bayesian neural networks minimise the Kullback-Leibler divergence to approximate the true posterior over the weights. However, this approximation is without knowledge of the final application, and therefore cannot guarantee optimal predictions for a given task. 
To overcome the challenge of making more suitable task-specific approximations, we introduce a new \emph{loss-calibrated} evidence lower bound for Bayesian neural networks in the context of supervised learning. By introducing a lower bound that depends on the utility function, we ensure that our approximation achieves higher utility than traditional methods for applications that have asymmetric utility functions. 
Through an illustrative medical example and a separate limited capacity experiment, we demonstrate the superior performance of our new loss-calibrated model in the presence of noisy labels. Furthermore, we show the scalability of our method to real world applications for per-pixel semantic segmentation on an autonomous driving data set.


## Example

![pedestrian_gain](https://github.com/AdamCobb/LCBNN/blob/master/models/segnet/exp4_150_epoch/plots/gain_ped_1.png)

## Getting Started

### Requirements
- [Python==3.6](https://www.python.org/getit/)
- [Tensorflow==1.6.0](https://www.tensorflow.org/)
- [keras==2.1.5](https://github.com/keras-team/keras/releases)
- [Jupyter](http://jupyter.org)

## Data
> The data for the SegNet experiment comes from the repository: https://github.com/alexgkendall/SegNet-Tutorial.git

## Contact Information
[Adam Cobb](https://adamcobb.github.io/): adam.cobb@sri.com

