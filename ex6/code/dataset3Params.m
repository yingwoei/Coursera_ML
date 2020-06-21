function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% Create a blank results matrix
results = zeros(length(C_vec) * length(sigma_vec), 3);

row = 1;
for C_val = C_vec
  for sigma_val = sigma_vec
    % Train 
    model = svmTrain(X, y, C_val, @(x1, x2) gaussianKernel(x1, x2, sigma_val));
    pred = svmPredict(model, Xval);
    % Compute validation set errors
    err_val = mean(double(pred ~= yval));
    % Save the results
    results(row,:) = [C_val sigma_val err_val];
    row = row + 1;
  endfor
endfor


% Find the minimum in the third column of resuls
[v i] = min(results(:,3));
% Retrieve C and sigma from the index
C = results(i, 1);
sigma = results(i, 2);






% =========================================================================

end
