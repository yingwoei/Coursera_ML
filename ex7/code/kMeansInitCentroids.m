function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

% create a matrix of only the unique rows
X_unq = unique(X, 'rows');

% Initialize the centroids to be random examples
% Randomly reorder the indices of examples
% create a random permutation of the rows  
randidx = randperm(size(X_unq, 1));

% take the first K rows as centroids
centroids = X_unq(randidx(1:K), :);
% =============================================================

end

