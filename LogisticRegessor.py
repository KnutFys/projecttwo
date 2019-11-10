# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:36:38 2019

@author: Knut H. Engvik
"""


import numpy as np
from time import time
from sklearn import datasets

class regressor:

    def __init__(self,X,y,eta=0.035):
        #Sets time limit for gradient decent
        self.time_limit = 3
        #Sets the iterations used in gradient decent
        self.iterations = 100
        #Set the learning rate
        self.eta = eta
        
        self.seed = 0
        
        #Set data variables
        #Feature matrix with shape numberofsamples times features
        self.X = X
        #The data attribute stores the target data
        #(Note to self: should probably rename this)
        self.data = y
        #Attributes to store testing data if data is split into
        #testing and training sets.
        self.test_data = y
        self.test_X = X
        #Sets classification threshold for single category regression
        self.threshold = 0.5
        #Set L1 and L2 regularisation
        self.L2 = 0
        self.L1 = 0
        #Sets a random weights
        self.beta = np.random.randn(len(X[0,:]))
        #initiate attributes to store classificationresults
        #---------for testing------------
        self.results = {}
        #----------------------------------
        self.classification = None
        self.training_classification = None
        self.prediction = None
        self.training_prediction = None
        self.classify()
        

    def gradient(self):
        #The gradient calculated from negative log-likelyhood cost function
        #is returned. The function also stores the estimated cost in a list.
        
        #Calculate gradient and store in list
        estimate = self.X@self.beta
        p = self.sigmoid(estimate)
        grad = -(self.X.T@(self.data-p))/len(self.data)
        #Add ridge penalty term
        grad = grad + (2*self.L2*self.beta)/len(self.data)        
        #Calculate cost and store in list        
        cost = (self.data.dot(estimate)-(np.log(1+np.exp(estimate))).sum())
        cost = -cost/len(self.data)
        
        #return gradient
        return grad
    
    def hessian(self):
        #Hessian calculated from negative log-likelyhood cost function
        #---------Not in use----------------------
        p = self.sigmoid(self.X@self.beta)
        W = np.eye(len(p))@(p-p^2)
        return self.X.T@W@self.X
    
    def sigmoid(self,x):
        return (1/(1+np.exp(-x)))
    
    def gradient_decent(self,verbose=False):
        t0 = time()
        if verbose:
            print("Running gradient decent. Number of iterations: {}".format(
                    self.iterations))
        for i in range(self.iterations):
            grad = self.gradient()
            self.beta -= self.eta*grad
            if (time()-t0) > self.time_limit:
                #Limits the time allowed for gradient decent dependent
                #on the time_limit attribute
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
    
    def confusion_matrix(self):
        c_matrix = np.zeros((2,2),int)
        for guess,target in zip(self.classification,self.test_data):
            if guess == 1:
                if target == 1: c_matrix[0,0] += 1
                if target == 0: c_matrix[0,1] += 1
            else:
                if target == 1: c_matrix[1,0] += 1
                if target == 0: c_matrix[1,1] += 1
        
        self.accuracy = np.trace(c_matrix)/np.sum(c_matrix)
        self.precision = c_matrix[0,0]/(c_matrix[0,0]+c_matrix[0,1])
        self.sensitivity = c_matrix[0,0]/np.sum(c_matrix[:,0])
        self.specificity = c_matrix[1,1]/(c_matrix[0,1]+c_matrix[1,1])
        self.baccuracy = (self.sensitivity+self.specificity)/2
        
        return c_matrix
    
    def classify(self):
        #This function updates the prediction attribute
        #using current values of weioghts(beta)
        self.prediction = self.sigmoid(self.test_X@self.beta)
        self.training_prediction = self.sigmoid(self.X@self.beta) 
        
    def evaluate(self,verbose= False):
        #Counts the number of correct predictions
        correct = 0
        #Sets proper length for classification attributes
        self.training_classification = np.zeros(len(self.data))
        self.classification = np.zeros(len(self.test_data))
        
        #Calculate testing score
        for ind,val in enumerate(self.prediction):
            self.classification[ind] = int(
                    1) if val > self.threshold else int(0)
        
        for pred,val in zip(self.classification,self.test_data):
            #print("Guess {} :: True {}".format(pred,val))
            if (pred == val): correct = correct + 1
        self.test_accuracy = correct/len(self.test_data)
        
        self.confusion_matrix()
        if verbose:
            print("Testing: Got {} out of {}".format(
                    correct,len(self.test_data)))   
            #print("Testing accuracy {:03.2f}".format(self.test_accuracy))
            print("Testing accuracy {:03.2f}".format(self.accuracy))
            print("Balanced accuracy {:03.2f}".format(self.baccuracy))
        #Calculate training score
        #Reset counter
        correct = 0
        for ind,val in enumerate(self.training_prediction):
            self.training_classification[ind] = int(
                    1) if val > self.threshold else int(0) 
        for pred,val in zip(self.training_classification,self.data):
            #print("Guess {} :: True {}".format(pred,val))
            if (pred == val): correct = correct + 1
        self.training_accuracy = correct/len(self.data)
        if verbose:
            print("Training: Got {} out of {}".format(correct,len(self.data)))
            print("Training accuracy {:03.2f}".format(self.training_accuracy))
            
    def reinitiate_weights(self):
        #Randomises weights for so testing can be performed with new
        #parameters
        self.beta = np.random.randn(len(self.X[0,:])) + 1
        
def normalize(x):
    return (x-np.amin(x))/(np.amax(x)-np.amin(x))
#iris  = datasets.load_iris()

def moon_test():
    #Simple test scikitlearn to
    print("Testing regressor on generated moon-dataset:")
    n = 1000
    X, y = datasets.make_moons(2*n, noise=0.2)
    #Create regressor instance and set training data
    t = regressor(X[:n-100],y[:n-100])
    #Set testing data
    t.test_X,t.test_data = X[n-100:],y[n-100:]
    t.eta = 0.3
    t.gradient_decent()
    #Count number of correct predictions using initial weights. The 
    #prediction used is 
    print("Result with random predictor weights:")
    t.evaluate(verbose=True)
    #Update regressor prediction with new weights
    t.classify()
    #Count number of correct predictions
    print("Result after gradient decent:")
    t.evaluate(verbose=True)   
    
def iris_test():
    import sklearn.datasets
    from sklearn.model_selection import train_test_split
    import pandas as pd
    #Testing regressor using Iris dataset for 2 classes
    iris = sklearn.datasets.load_iris()
    iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
    y = iris.target
    intercept = np.ones(len(y))
    columns = iris_df.columns    
    X = np.c_[intercept,iris_df[columns]]
    X = normalize(X)
    X = X[:100,:]
    y = y[:100]
    X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)    
    r2 = regressor(X_train,y_train)
    r2.test_data = y_test
    r2.test_X = X_test
    r2.iterations = 100
    r2.eta = 1
    print("Testing regressor on Iris dataset. Iterations: {}".format(
            r2.iterations))
    print("Random guess:")
    r2.classify()
    r2.evaluate(verbose=True)
    r2.gradient_decent()
    print("After training guess:")
    r2.classify()
    r2.evaluate(verbose=True)
def main():
    moon_test()
    iris_test()

if __name__ == "__main__":
    main()
