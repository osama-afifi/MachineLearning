function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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


trials = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];

iteration = 1;
bestPrediction = 1; % Worsr Case
for CTrial = trials
	for sigmaTrial = trials
			fprintf(['Trial #%d: C = %f, sigma = %f\n'], iteration, CTrial, sigmaTrial);
			iteration = iteration + 1;
			model = svmTrain(X, y, CTrial, @(x1, x2) gaussianKernel(x1, x2, sigmaTrial)); % Train an SVM Model (get Theta)
			predictions = svmPredict(model, Xval); % Compute Predictions based on thr Model
			predictionsError = mean(double(predictions ~= yval)); % Compute Error with the CrossValidation data
			fprintf(['Error is  %f\n'], predictionsError);
			if(predictionsError < bestPrediction)
				fprintf(['Current Best Prediction Error %f\n'], predictionsError);
				bestPrediction = predictionsError;
				sigma = sigmaTrial;
				C = CTrial;
			end;
	end;
end;
fprintf(['\nBest Error Prediction  %f\n Best Sigma Value is %f\n Best C Value is %f\n'], bestPrediction, sigma, C);

% =========================================================================

end
