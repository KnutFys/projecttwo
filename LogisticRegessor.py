# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:36:38 2019

@author: Knut H. Engvik
"""


import numpy as np
from time import time
from sklearn import datasets
from sklearn import linear_model

import matplotlib.pyplot as plt



class regressor:

    def __init__(self,X,y,eta=0.001):
        #Sets time limit for gradient decent
        self.time_limit = 3
        #Sets the iterations used in gradient decent
        self.iterations = 50
        #Set the learning rate
        self.eta = eta
        
        self.seed = 0
        
        #Set data variables
        self.X = X
        self.data = y
        self.test_data = y
        self.test_X = X
        
        #Sets a random weights
        self.beta = np.random.randn(len(X[0,:]))
        self.counter=0
        self.classification = None
        self.training_classification = None
        self.prediction = None
        self.training_prediction = None
        self.classify()
        
    def gradient(self,p):
        #The gradient calculated from 
        return -(self.X.T@(self.data-p))
    
    def hessian(self,p):
        W = np.eye(len(p))@(p-p^2)
        return self.X.T@W@self.X
    
    def sigmoid(self,x):
        return (1/(1+np.exp(-x)))
    
    def gradient_decent(self):
        t0 = time()
        for i in range(self.iterations):
            self.counter = self.counter+i
            self.beta -= self.eta*self.gradient(self.sigmoid(self.X@self.beta))
            if (time()-t0) > self.time_limit:
                print("Time limit reached.")
                print("Running time: {}".format(time()-t0))
                print("Gradient decent might not have converged")
                print("Performing evaluation step...")
                print("Current value of predictors:")
                print(self.beta)
                print("Next step value of predictors:")
                print(self.beta-(self.eta*self.gradient(
                        self.sigmoid(self.X@self.beta))))                
                break
            
    def classify(self):
        #This function updates the prediction attribute
        #using current values of weioghts(beta)
        self.prediction = self.sigmoid(self.test_X@self.beta)
        self.training_prediction = self.sigmoid(self.X@self.beta) 
        
    def evaluate(self):
        #Counts the number of correct predictions
        correct = 0
        #Sets proper length for 
        self.training_classification = np.zeros(len(self.data))
        self.classification = np.zeros(len(self.test_data))
        
        #Calculate testing score
        for ind,val in enumerate(self.prediction):
            self.classification[ind] = int(1) if val > 0.5 else int(0) 
        for pred,val in zip(self.classification,self.test_data):
            #print("Guess {} :: True {}".format(pred,val))
            if (pred == val): correct = correct + 1 
        print("Testing: Got {} out of {}".format(correct,len(self.test_data)))   
        print("Testing accuracy {:03.2f}".format(correct/len(self.test_data)))
        #Calculate training score
        correct = 0
        for ind,val in enumerate(self.training_prediction):
            self.training_classification[ind] = int(1) if val > 0.5 else int(0) 
        for pred,val in zip(self.training_classification,self.data):
            #print("Guess {} :: True {}".format(pred,val))
            if (pred == val): correct = correct + 1
        print("Testing: Got {} out of {}".format(correct,len(self.data)))
        print("Training accuracy {:03.2f}".format(correct/len(self.data))) 


#iris  = datasets.load_iris()

def moon_test():
    #Simple test scikitlearn to
    n = 1000
    X, y = datasets.make_moons(2*n, noise=0.2)
    #Create regressor instance and set training data
    t = regressor(X[:n-100],y[:n-100])
    #Set testing data
    t.test_X,t.test_data = X[n-100:],y[n-100:]
    t.gradient_decent()
    #Count number of correct predictions using initial weights. The 
    #prediction used is 
    print("Result with random predictor weights:")
    t.evaluate()
    #Update regressor prediction with new weights
    t.classify()
    #Count number of correct predictions
    print("Result after gradient decent:")
    t.evaluate()   

def main():
    moon_test()

if __name__ == "__main__":
    main()
