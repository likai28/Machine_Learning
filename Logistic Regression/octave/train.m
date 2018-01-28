function weight_train = train(X, Y, maxNumIters, alpha, regLambda, epsilon)
%Train the logistic regression model using gradient descent
%Input: X (n-by-d matrix), Y (n-by-1 matrix), maxNumIters, 
%alpha, regLambda, epsilon
%output: the updated weight ((d+1)-by-1 matrix)

%Add the 1's column
[n,d] = size(X);
newX = [ones(n,1),X];

%Initialize the weight as zeros
weight_train = zeros(d+1,1);


end