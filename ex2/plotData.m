function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


% Find Indices of Positive (y=1) and Negative (y=0) class
pos = find(y==1);
neg = find(y==0);

% Plot points with admission, positive class.
% X(pos,1) is the first value, ie: exam 1 score;
% X(pos,2) is the second value, i.e: exam 2 score
% k+, black cross hair
plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);

% X(neg, 1) is exam 1 score for points with no admission
% X(neg, 2) is exam 2 score for points with no admission
% ko, black circle, filled yellow, 7 for size
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
ylabel('Exam 2 score');
xlabel('Exam 1 score');









% =========================================================================



hold off;

end
