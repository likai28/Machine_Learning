#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 50):
        '''
        Initializes Parameters of the  Logistic Regression Model
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
  
    

    
    
    def calculateGradient(self, weight, X, Y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
        
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is (d+1)-by-1 dimensional numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an (d+1)-by-1 dimensional numpy matrix
        '''
        n = X.shape[0]
        Z = np.dot(X,weight)
        sigmoid = self.sigmoid(Z)
        Gradient = np.ones((X.shape[1],1))
        Gradient[0] = np.sum((sigmoid-Y)*X[:,0].reshape(n,1),axis=0)

        tmp = np.sum((sigmoid-Y)*X[:,1:],axis=0).reshape(X.shape[1]-1,1) + regLambda*weight[1:]
        tmp = tmp.reshape(X.shape[1]-1,1)
        Gradient[1:] = tmp

        
        return Gradient    

    def sigmoid(self, Z):
        '''
        Computes the Sigmoid Function  
        Arguments:
            A n-by-1 dimensional numpy matrix
        Returns:
            A n-by-1 dimensional numpy matrix
       
        '''
        sigmoid = 1/(1+np.exp(-Z))
        
        return sigmoid

    def update_weight(self,X,Y,weight):
        '''
        Updates the weight vector.
        Arguments:
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is a d+1-by-1 dimensional numpy matrix
        Returns:
            updated weight vector : (d+1)-by-1 dimensional numpy matrix
        '''
        new_weight = weight - self.alpha*self.calculateGradient(weight,X,Y,self.regLambda)
        
        
        return new_weight
    
    def check_conv(self,weight,new_weight,epsilon):
        '''
        Convergence Based on Tolerance Values
        Arguments:
            weight is a (d+1)-by-1 dimensional numpy matrix
            new_weights is a (d+1)-by-1 dimensional numpy matrix
            epsilon is the Tolerance value we check against
        Return : 
            True if the weights have converged, otherwise False

        '''
        errors = np.sum(np.square(new_weight-weight),axis=0)[0]
        errors = np.sqrt(errors)
        if errors<= epsilon:
            return True
        
        return False
        
    def train(self,X,Y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            Y is an n-by-1 dimensional numpy matrix
        Return:
            Updated Weights Vector: (d+1)-by-1 dimensional numpy matrix
        '''
        # Read Data
        n,d = X.shape
        
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        self.weight = self.new_weight = np.zeros((d+1,1))

        for i in range(self.maxNumIters):
            self.new_weight = self.update_weight(X,Y,self.weight)
            if self.check_conv(self.weight,self.new_weight,self.epsilon)==True:
                self.weight = self.new_weight
                break
            else:
                self.weight = self.new_weight
         
        return self.weight

    def predict_label(self, X,weight):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
            weight is a d+1-by-1 dimensional matrix
        Returns:
            an n-by-1 dimensional matrix of the predictions 0 or 1
        '''
        #data
        n=X.shape[0]
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        Z = np.dot(X,weight)
        result = np.round(self.sigmoid(Z))
        result = result.reshape(n,1)
        
        return result
    
    def calculateAccuracy (self, Y_predict, Y_test):
        '''
        Computes the Accuracy of the model
        Arguments:
            Y_predict is a n-by-1 dimensional matrix (Predicted Labels)
            Y_test is a n-by-1 dimensional matrix (True Labels )
        Returns:
            Scalar value for accuracy in the range of 0 - 100 %
        '''
        n= Y_test.shape[0]
        correct =0.0
        for i in range(n):
            if Y_predict[i][0]==Y_test[i][0]:
                correct=correct+1

        Accuracy = (correct/n)*100
        return Accuracy
    
        