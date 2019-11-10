# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:35:15 2019

@author: Knut Engvik
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from time import time

class network:
    def __init__(self,structure,smoosh_weights=True):
        
        #initiate random weights and biases based on the structure
        #argument. IE a tuple (a,b,c) will have an input
        #layer of size a, one hidden layer of size b and
        #an output layer of size c
        self.number_of_layers = len(structure)        
        self.bias = [np.random.randn(y,1) for y in structure[1:]]
        if smoosh_weights:
            self.weights = [np.random.randn(x,y)/np.sqrt(y) for x,y in zip(
                    structure[1:],structure[:-1])]
        else:
            self.weights = [np.random.randn(x,y) for x,y in zip(
                    structure[1:],structure[:-1])]
        #set the training rate
        self.learning_rate = 0.01
        #Add an attribute to determine what cost function to use
        self.cost_function = None
        #Store the shape/structure of the network
        self.structure = structure
        #Set the levelel of L2 regularization
        self.L2 = 0
        #Set threshold for prediction in cases with single output
        #category. Lower threshold increases likelyhood of being classified
        #in the single category
        self.threshold = 0.5
        #Determines whether the network is a classifier or tries to fit 
        #a function
        self.fit_function = False
        
    #Define activation function and its derivative
    #Here a sigmoid is used
    def sigmoid(self,x):
        return (1/(1+np.exp(-x)))
    
    #Not used ATM ---------------------!!!!!
    def sigmoid_derivative(self,x):
        print("Placeholder function" )
    
    #Softmax if more than one category is  used in classification
    #Not used ATM ---------------------!!!!!
    def softmax(self, x):
        return np.exp(x)/np.exp(x).sum()
    
    def deltaL(self,target,a,da):
        #Returns the output error for variuous choices of cost function.
        #Default cost function is cross entropy
        #Currently the arguments are target: the target data/correct answers
        # a: is the activations at the output layer
        # da: is the derivative of a with respect to z where z is
        # the input from the previous layer
        if self.cost_function == "square":
            #If quadratic cost function is used
            return ((a-target)*da)
        
        else:
            #If cross entropy is used. This is the default
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
        #Prepare to store gradients
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #Using sigmoid activation functions for forward propagation
        activations,d_activations = self.forward_sigmoid(x)
        #Calculate output error
        delta = self.deltaL(y,activations[-1],d_activations[-1])
        #Store output layer gradients
        nabla_b[-1] += delta.sum(axis=1,keepdims= True)/n
        nabla_w[-1] += np.einsum("kn,jn->kj",delta,activations[-2])/n
        #Add regularization penalty
        nabla_w[-1] += (self.L2/n)*self.weights[-1]
        #Calculate remaining gradients for the hidden layers
        for index in range(2,self.number_of_layers):
            delta = (self.weights[-index+1].T@delta)*d_activations[-index]
            nabla_b[-index] += delta.sum(axis=1,keepdims= True)/n
            nabla_w[-index] += np.einsum(
                    "kn,jn->kj",delta,activations[-index-1])/n
            #Add regularization penalty
            nabla_w[-index] += (self.L2/n)*self.weights[-index]

        #Loop through all the calculated gradients and 
        #update biases and weights
        for index, (b,w) in enumerate(zip(nabla_b,nabla_w)):
            self.bias[index]=self.bias[index]-(self.learning_rate*b)
            self.weights[index]=self.weights[index]-(self.learning_rate*w)
        

    def train(self, data, target):
        #ATM not necessary.
        #Placeholder function in case the need arises to split
        #the backpropagation function
        self.back(data,target)
        
    def test(self,x,y):        
        #NOTE: single categories need to be handled differently
        #Use sigmoid neurons fo forward propagation
        z,dz = self.forward_sigmoid(x)
        #Check to see if the network is trying to fit a function
        #or classify data
        if self.fit_function:
            return z
        teller = 0
        #For single/binary class classification, No onehot needed
        if self.structure[-1] == 1:
            #Create empty confusion matrix
            confusion_m = np.zeros((2,2),int)
            n = len(y)
            for activation,answer in zip(z[-1][0,:],y):
               guess = 1 if activation > self.threshold else 0
               guessv = np.array([guess, 1-guess])
               answerv = np.array([answer, 1 -answer])
               confusion_m += np.outer(guessv,answerv)
               if guess == answer: teller += 1 
        else:
            confusion_m = np.zeros((len(y[:,0]),len(y[:,0])),int)
            n = len(y[0,:])
            for i in range(n):
                #Create a onehot representation of the prediction vector
                guessv = np.zeros(len(y[:,0]),int)
                guess =np.argmax(z[-1][:,i])
                guessv[guess] = 1
                #Find target/answer by finding the index of the 
                #one in the one hot target vector
                ansv = np.argmax(y[:,i])
                #Update confusion matrix
                #print("------------------------")
                #print(np.outer(guess,y[:,i]))
                confusion_m += np.outer(guessv,y[:,i])
                if ansv == guess:
                    #Counts number of "hits"
                    teller += 1
        #print("Got {} out of {}".format(teller,n))
        return teller/n, confusion_m

    def make_batch_index(self,n):
        #Returns a randomized index list for use in 
        #creating mini batches.
        return np.random.permutation(n)

    def visualize(self,x,y,x2=None,y2=None):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(x,y)
        ax2.plot(x,y2)
        #fig.show()
     
    def set_bias(self):
        self.bias = [np.ones((y,1)) for y in self.structure[1:]]
        
def digit_test():
    #Load data. NOTE DATA is transposed compared to expected input
    #--------------------------------
    #Testing Network on handwritten digit recognition
    digits = sklearn.datasets.load_digits()
    y = digits.target
    x = digits.data
    #Create onehots of the target
    ohy = np.zeros((10,len(y)),dtype=int)
    for index , y_value in enumerate(y):
        ohy[y_value,index] = 1
    
    #split data in training testing sets
    x_train,x_test,y_train, y_test = train_test_split(x,ohy.T,test_size=0.2)
    #Format so data fits network, ie transpose it
    x_train = x_train.T
    x_test = x_test.T
    y_train = y_train.T
    y_test = y_test.T
    #initiate network
    net = network((64,100,10))
    #net = network((64,256,128,10))
    #Choose cost function
    #net.cost_function = "square"
    #Set the batch size 
    batch_size = 10
    #Set number of training epochs
    number_of_epochs = 100
    #set the training rate
    net.learning_rate = 0.1
    #Set regularization
    net.L2 = 0.0001   
    #Creates lists to store training and testing accuracy
    training_results = list()
    test_results = list()
    confusion_test = None
    confusion_train = None
    #Create time stamp to track calculation times
    t0 = time()
    for e in range(number_of_epochs):
        #Keep track of raining progress and time usage
        print("\rRunning time after {} epochs: {}".format(e,time()-t0),
              end="",flush=True)
        #Add results for current weights, biases
        #On testing set
        accuracy ,confusion_test = net.test(x_test,y_test)
        test_results.append(accuracy)
        #On training set
        accuracy ,confusion_train = net.test(x_train,y_train)
        training_results.append(accuracy)
        #Ceate a randomized index vector k
        k = net.make_batch_index(len(x_train[0,:]))
        #Calculate number of batches and loop through
        for i in range(len(k)//batch_size):
            batch = k[i*batch_size:(i+1)*batch_size]
            net.train(x_train[:,batch],
                              y_train[:,batch])
    print("")
    print("Training accuracy: {} Testing accuracy: {}".format(
            training_results[-1],test_results[-1]))
    net.visualize(np.arange(number_of_epochs),training_results,
                  y2=test_results)        
    print(confusion_test)
    print(confusion_train)
def main():
    #Run test classifying handwritten digits
    digit_test()

if __name__ == "__main__":
    main()