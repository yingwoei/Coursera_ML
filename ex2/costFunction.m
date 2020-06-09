function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% Compute h using sigmoid function created earlier.
z = X * theta;
h = sigmoid(z);

% Vector multiplication, i.e.: dot product, will give you sum of the products
term1 = (-y)' * (log(h)); % first term is a scalar from multiplying 1 x m  and m x 1.
term2 = (1-y)' * (log(1-h)); % second term is also scalar for the same reason.
J = (1/m) * (term1 - term2); % Unregularized cost function

% Compute gradient
% Transpose X to get n x m since h-y is m x 1
error = h - y;
grad = (1/m) * (X' * error);

% =============================================================

end