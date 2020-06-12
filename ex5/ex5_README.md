## What I learned from Assignment 5
### Regularized Linear Regression & Bias VS Variance

* To improve a learning algorithm, datasets should be splits into 3 sets - training set, cross-validation set, and test set.
  * It's crucial to shuffle the data randomly before splitting.
* Use training set to determine theta and polynomial degree for regression.
* Use cross-validation set to evaluate and determine the optimal regularization parameter, lambda.
  * The ideal lamba will be the one that gives the lowest cross-validation error.
* Use test set to determine the performance of an algorithm.
* A model without regularization fits the training set but does not generalize well, so can work poorly on test cases.
* It's important to not regularize the cost function twice. If regularization is applied to determine theta, there is no need to reapply regularization on cost function. So we should set lambda = 0.
