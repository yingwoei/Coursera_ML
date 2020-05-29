## What I learned from Assignment 2
* How to compute logistic regression - cost function, gradient, and hypothesis.
* How lambda helps to regularize regression to reduce overfitting.
* Regularization is an algorithm that shrinks all parameters theta.
* theta(1) in octave is theta_0 in math because Ocatve indexing starts at 1.
* theat(1) = 0 to exclude the bias feature.
* This is because Regularization puts a penalty on large (non-bias) theta values. This makes hypotheses less likely to overfit. The purpose of regularization is to reduce overfitting by making the weights smaller. Since large bias causes underfitting rather than overfitting, we don’t want to reduce the bias when we are overfitting. For this reason the “bias term” weights don’t participate in regularization.
* Hypothesis in logistic regression measures the probability of one test case belong to Class 0/1.
