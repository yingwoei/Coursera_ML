function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
% Refer to ex1

% Cost function
h = X * theta; %Hypothesis or prediction of training set X is mxn, theta is nx1, h will be mx1
errs = h - y; % errs is m x 1
theta(1) = 0; % to exclude bias feature from regularization, theta(1) should still be considered above
reg = (lambda/(2*m)) * sum(theta.^2);
reg_grad = (lambda/m) * theta; 

J = ((1/(2*m)) * (errs'*errs)) + reg;
grad = (1/m)*(X' * errs) + reg_grad; % X is m x n, you get n x 1 for grad

% =========================================================================

grad = grad(:);

end
