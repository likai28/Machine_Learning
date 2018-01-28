%Load the data
XTrain = csvread('../data/XTrain.csv');
yTrain = csvread('../data/YTrain.csv');
XTest = csvread('../data/XTest.csv');
yTest = csvread('../data/YTest.csv');

%check the sizes of the data
%disp(size(X_Train));
%disp(size(Y_Train));
%disp(size(X_Test));
%disp(size(Y_Test));

%Set the parameters
maxNumIters = 50;
alpha = 0.01;
regLambda = 0.01;
epsilon = 0.0001;

%Train the model
LogReg = train(XTrain, yTrain, maxNumIters, alpha, regLambda, epsilon);

%Predict labels for test data
yPredict = predict_label(XTest, LogReg);

%Calculate accuracy of the model on test data
accuracy = calculateAccuracy(yPredict, yTest);

