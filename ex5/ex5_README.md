## What I learned from Assignment 5
### Regularized Linear Regression & Bias VS Variance

* To improve a learning algorithm, datasets should be splits into 3 sets - training set, cross-validation set, and test set.
  * It's crucial to shuffle the data randomly before splitting.
* Use training set to determine theta and polynomial degree for regression, J_train.
* Use cross-validation set to evaluate (find error J_cv) and determine the optimal regularization parameter, lambda.
  * The ideal lamba will be the one that gives the lowest cross-validation error.
* Use test set to determine the performance of an algorithm with theta and lambda determine from training and cross-validation set.
* A model without regularization fits the training set but does not generalize well, so can work poorly on test cases.
* It's important to not regularize the cost function twice. If regularization is applied to determine theta, there is no need to reapply regularization on cost function. So we should set lambda = 0.

#### To determine test set error:
* Once an ideal lambda is determined from cross validation set, you'd go back and retrain theta on the training set specifying the desired lambda = k.
* Use the theta and lambda = 0 to compute error_test. 
