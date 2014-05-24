function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Theta1 : 25 x 401
% Theta2 : 10 x 26

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Remember if network has s(j) units in level j
% and s(j+1) units in level j+1
% then the resulting theta(j) is of dimension --->  s(j+1) x s(j) +1

a1 = [ones(m, 1) X]; % 5000 * 401

z2 = a1*Theta1';  % 5000x25
a2 = sigmoid(z2); % 5000x25
a2 = [ones(m, 1) a2]; %5000x26
z3 = a2*Theta2'; %5000x10
a3 = sigmoid(z3); %5000x10

[maximum, p] = max(a3,[],2);
% p is 5000x1










% =========================================================================


end
