% The NB_XGivenY function takes a training set XTrain and yTrain and
% Beta parameters beta_0 and beta_1, then returns a matrix containing
% MAP estimates of theta_yw for all words w and class labels y
function [D] = NB_XGivenY(XTrain, yTrain, beta_0, beta_1)
    %% Inputs %% 
    % XTrain - (n by V) matrix
    % yTrain - 1D vector of length n
    % alpha - scalar
    % beta - scalar

    %% Outputs %%
    % D - (2 by V) matrix

    D = zeros(2, size(XTrain,2));
end