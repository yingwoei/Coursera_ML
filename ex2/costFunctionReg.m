function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% don't regularize first theta, theta(1), i.e.: theta-zero
% Compute h using sigmoid function created earlier.
z = X * theta;
h = sigmoid(z);

% Vector multiplication, i.e.: dot product, will give you sum of the products
term1 = (-y)' * (log(h)); % first term is a scalar from multiplying 1 x m  and m x 1.
term2 = (1-y)' * (log(1-h)); % second term is also scalar for the same reason.
theta(1) = 0; % to exclude bias feature

reg_cost = (lambda/(2*m)) * (theta'*theta); % Regularization cost
J = ((1/m) * (term1 - term2)) + reg_cost; % Regularized cost function

% Compute gradient
error = h - y;
reg_grad = (lambda/m) * theta;
grad = ((1/m) * (X' * error)) + reg_grad; % Transpose X to get n x m since h-y is m x 1


% =============================================================

end
