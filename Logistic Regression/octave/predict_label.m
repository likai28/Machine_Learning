function result = predict_label(XTest, weight_train)
%Use the fitted model to predict values for each instance in XTest
%Input: XTest(n-by-d matrix), weight_train((d+1)-by-1 matrix)
%Output: result(n-by-1 matrix of the predictions)

%Add the 1's column
[n,d] = size(XTest);
newXTest = [ones(n,1),XTest];



end
