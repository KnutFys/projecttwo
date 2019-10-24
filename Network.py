# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:35:15 2019

@author: Knut Engvik
"""

import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from time import time

class network:
    def __init__(self,structure):
        
        #initiate random weights and biases based on the structure
        #argument. IE a tuple (a,b,c) will have an input
        #layer of size a, one hidden layer of size b and
        #an output layer of size c
        self.number_of_layers = len(structure)
        self.bias = [np.random.randn(y,1) for y in structure[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(
                structure[1:],structure[:-1])]
        #set the training rate
        self.learning_rate = 0.01
        self.cost_function = None
    #Define activation function and its derivative
    #Here a sigmoid is used
    def sigmoid(self,x):
        return (1/(1+np.exp(-x)))
    
    #Not used ATM ---------------------!!!!!
    def sigmoid_derivative(self,x):
        return
    
    #Softmax if more than one category is  used in classification
    def softmax(self, x):
        return np.exp(x)/np.exp(x).sum()
    
    def deltaL(self,target,a,da):
        #The gradient calculated from logistic cost function
        #
        if self.cost_function == "square":
            #If quadratic cost function is used
            print("Using quadratic cost function")
            
            return ((a-target)*da)
        elif self.cost_function == "ridge":
            #If ridge is used 
            print("Using ridge")
        else:
            #If cross entropy is used 
            a = (a-target)
            return a
        
    def forward_sigmoid(self,a):
        
        #Create lsits to store activations and derivatives of activation
        #functions using sigmoid as activation function
        z = list()
        dz = list()
        #Add input in list of activations
        z.append(a)
        for b,w in zip(self.bias,self.weights):
            #Calculate next activation
            a = self.sigmoid(w@a+b)
            #Store activation in list
            z.append(a)
            #Calculate and store derivative of activation
            #in list
            dz.append(a*(1-a))
        return z,dz
        #return [self.sigmoid(w@a+b) for b,w in zip(self.bias,self.weights)]
    
    def back(self,x,y):
        #Forward propagation of input matrix
        #Format must be [x1 x2..] wher xi are 
        #data points/training vectors i.e 
        #dimensions are (features,number of datapoints)
        #All activations and derivatives of activation
        #functions are returned
        
        #Number of input vectors
        n = len(x[0,:])
        #Using sigmoid activation functions
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activations,d_activations = self.forward_sigmoid(x)
        #Calculate output error
        delta = self.deltaL(y,activations[-1],d_activations[-1])
        #Store output layer gradients
        nabla_b[-1] += delta.sum(axis=1,keepdims= True)/n
        nabla_w[-1] += np.einsum("kn,jn->kj",delta,activations[-2])/n
        #Calculate remaining gradients for the hidden layers
        for index in range(2,self.number_of_layers):
            delta = (self.weights[-index+1].T@delta)*d_activations[-index]
            nabla_b[-index] += delta.sum(axis=1,keepdims= True)/n
            nabla_w[-index] += np.einsum(
                    "kn,jn->kj",delta,activations[-index-1])/n

        #Loop through all the calculated gradients and 
        #update biases and weights
        for index, (b,w) in enumerate(zip(nabla_b,nabla_w)):
            self.bias[index]=self.bias[index]-(self.learning_rate*b)
            self.weights[index]=self.weights[index]-(self.learning_rate*w)
        

    def train(self, data, target):
        self.back(data,target)
        
    def test(self,x,y):
        #Number of test inputs
        n = len(y[0,:])
        z,dz = self.forward_sigmoid(x)       
        teller = 0
        for i in range(n):
            a,b = np.argmax(z[-1][:,i]),np.argmax(y[:,i])
            #print("Guess: {} Answer:{}".format(a,b))
            if a == b:
                #print("Huzza!")
                teller += 1
        print("Got {} out of {}".format(teller,n))
        return teller/n

def digit_test():
    #Load data. NOTE DATA is transposed compared to expected input
    #--------------------------------
    digits = sklearn.datasets.load_digits()
    y = digits.target
    x = digits.data
    #Create onehots of the target
    ohy = np.zeros((10,len(y)),dtype=int)
    for index , y_value in enumerate(y):
        ohy[y_value,index] = 1
    
    #split data in training testing sets
    x_train,x_test,y_train, y_test = train_test_split(x,ohy.T,test_size=0.2)
    #Format so data fits network ie transpose it
    x_train = x_train.T
    x_test = x_test.T
    y_train = y_train.T
    y_test = y_test.T
    #initioate network
    net = network((64,256,128,10))
    
    def make_batch_index(n):
        return np.random.permutation(n)
    
    t0 = time()
    
    number_of_epochs = 1000
    results = list()
    for e in range(number_of_epochs):
        print("After {} epochs.....".format(e))
        print(time()-t0)
        results.append(net.test(x_test,y_test))
        number_of_batches = 5
        k = make_batch_index(len(x_train[0,:]))
        batch_size = len(k)//number_of_batches 
        for i in range(number_of_batches):
             net.train(x_train[:,i*batch_size:(i+1)*batch_size],
                               y_train[:,i*batch_size:(i+1)*batch_size])


