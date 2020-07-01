function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K, # of centroids
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1); % 300 x 1 vector

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


m = size(X, 1); % # of training examples
% --------- Without bsxfun and sum of square ------------------

##for i = 1:m
##  d = zeros(K, 1);
##  for j = 1:K
##    d(j) = sqrt(sum((X(i,:) - centroids(j,:)).^2));
##  endfor
##  [~, idx(i)] = min(d);
##endfor


% --------- Tutorial's approach -----------------------
% Create a "distance" matrix of size (m x K)
% initialize it to all zeros.
distance = zeros(m, K); % 300 x 3 matrix

% iterate through centroids instead of training examples because it's faster.
for i = 1:K
  temp = bsxfun(@minus, X, centroids(i,:));
  distance(:, i) = sum(temp.^2,2);
endfor

[dummy, idx] = min(distance, [], 2);


%[val idx] = min(distance); 
% recall min() gives you 2 outputs
% min(X,[],2) is a column vector gives you minimum of every row/dimension requested
% min(X) gives you minimum of every column
% Refer to: https://www.mathworks.com/help/matlab/ref/min.html
% [val, idx] = min(X) gives you minimum of every column and its index

% =============================================================

end

