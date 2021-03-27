# OODISH
Pytorch implementation of Out-Of-Distribution Error Detection algorithms

This project is primarily divided into two IronPython notebooks which can be loaded using Jupyter or your interpreter of choice.

The first portion, autoencoder-pytorch_validate.ipynb loads a given dataset, constructs the CAE model with the given parameters, then runs the parameter sets until the specified number of epochs elapses or the early stopping criteria is met, or until memory runs out.
It will save the best-performing model in the dataset folder and display some basic reconstructions so we can demonstrate it is working.

The second portion, load-anomaly-testing.ipynb will load the given model from the first notebook, perform some initial collection of distribution data for comparative purposes, then generate a specified number of test cases with a 50% chance of a given test case being adversarial.
The model will then be used to reconstruct the image, the loss measured using MSE, SSIM and the encoding divergence measured using Mahalanobis distance, the ROC curve and accuracy scores will be generated and the loss comparison scatterplot and reconstruction histograms generated.
Finally the model will output the first 10 images and their encodings and reconstructions, then all of the failures (FNs and FPs) for examination.
