function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

y_matrix = eye(num_labels)(y,:); %y_matrix is 5000 x 10
X = [ones(m, 1) X]; %include bias unit
a1 = X; %a1 is 5000 x 401, Theta1 is 25 x 401, looking at ex4weights.mat
z2 = a1 * (Theta1)'; %z2 is 5000 x 25
a2 = sigmoid(z2); %a2 is 5000 x 25
a2_m = size(a2, 1); % get dimension of rows of a2
a2 = [ones(a2_m, 1) a2]; %include bias unit in the first column, a2 is now 5000 x 26
z3 = a2 * (Theta2)'; %z3 is 5000 * 10, Theta2 is 10 x 26
a3 = sigmoid(z3); %a3 is h, our hypothesis

% y_matrix and h/a3 are of size m x k 
% don't calculate with y! y_matrix instead
% Use element-wise multiplication for natural log
term1 = (-y_matrix)' * log(a3); %k x k
term1 = trace(term1); % to get the sum of the main diagonal
term2 = (1 - y_matrix)' * log(1-a3); % k x k
term2 = trace(term2);
J = (1/m)*(term1 - term2);

% J with regularization
% exclude the bias unit from regularization by using Theta1(:,2:end)
reg_term1 = Theta1(:,2:end) * Theta1(:,2:end)'; %Theta1 is 25 x 401
reg_term1 = trace(reg_term1);
reg_term2 = Theta2(:,2:end) * Theta2(:,2:end)'; %Theta2 is 10 x 26
reg_term2 = trace(reg_term2);
reg_term = (lambda/(2*m))*(reg_term1 + reg_term2);
J = (1/m)*(term1 - term2) + reg_term; 

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% m = # of training examples
% n = # of training features, including the initial bias unit
% h = # of units in the hidden layer, NOT including bias unit
% r = # of output classification

% Forward Propagation
y_matrix = eye(num_labels)(y,:); %y_matrix is 5000 x 10
% X = [ones(m, 1) X]; % don't need this line as X includes bias unit already from previous execution.
a1 = X; %a1 is 5000 x 401, Theta1 is 25 x 401, looking at ex4weights.mat
z2 = a1 * (Theta1)'; %z2 is 5000 x 25, m x h
a2 = sigmoid(z2); %a2 is 5000 x 25
a2_m = size(a2, 1); % get dimension of rows of a2
a2 = [ones(a2_m, 1) a2]; %include bias unit in the first column, a2 is now 5000 x 26
z3 = a2 * (Theta2)'; %z3 is 5000 * 10, Theta2 is 10 x 26
a3 = sigmoid(z3); %a3 is h, our hypothesis

% Back Propgation
d3 = a3 - y_matrix; % a3 and y_matrix is 5000 x 10 / m x r
g_z2 = sigmoidGradient(z2); 
d2 = d3 * Theta2(:,2:end) .* g_z2; % d2 is m x h, 5000 x 25, same size as z2
Delta1 = d2' * a1; %Delta1 is 25 x 401
Delta2 = d3' * a2; % Delta2 is r x (h+1) 10 x 26
Theta1_grad = (1/m) * Delta1; % 25 x 401
Theta2_grad = (1/m) * Delta2;

% Theta gradient with regularization
% bias unit should not be regularized
% Set the first column of Theta1 and Theta2 to all-zeros so that the dimension fits
%if you use Theta1(:,2:end) to calculate Theta1_grad, Theta1 will become 25 x 400, dimension mismatch

Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1_grad = (1/m) * Delta1 + (lambda/m)*Theta1;
Theta2_grad = (1/m) * Delta2 + (lambda/m)*Theta2; 

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
