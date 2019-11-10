# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 00:22:58 2019

@author: Knut H Engvik
"""
import numpy as np
from time import time
import pandas as pd
from Network import network
t0 = time()
#Set up functions
def make_data(datapoints):
    #This function  makes a "coordinate system"
    #that the Frankefunction can be evaluated on. 
    #It takes an integer valued argument that defines the coarseness of
    #the grid.
    a = np.arange(0, 1, 1/datapoints)
    b = np.arange(0, 1, 1/datapoints)
    x, y = np.meshgrid(a,b)
    return x, y

def franke_function(x,y, noiceLevel=0):
    #This function calculates the Franke function on the meshgrids x, y
    #It then adds noice, drawn from a normal distribution, with mean 
    #0, std  noiceLevel, to each calculated datapoint of 
    # the Franke function.
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    
    if noiceLevel != 0:
        np.random.seed()
        noice = np.random.randn(x.shape[0],x.shape[1])*noiceLevel
        return term1 + term2 + term3 + term4 , noice
    else:
        noice = np.zeros(x.shape)
        return term1 + term2 + term3 + term4 , noice

def design_matrix(x,y):
    #This functions sets up a design matrix ie a set pairs (x,y) that
    #correspods to the coordinates for frankes function
    DM = np.zeros((len(x),2))
    DM[:,0] = x
    DM[:,1] = y
    return DM


def R2(y_data, y_model):
    #This function calculates and returns the R2 value between a 
    #
    return 1-np.sum((y_data - y_model)**2)/np.sum((y_data-np.mean(y_data))**2)



def MSE(y_data,y_model):
    #This function calculates and returns the mean squared error 
    #between y_data and y_model
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def grid_search(learning_rates,lambdas,hidden,n_epochs=[5],b_sizes=[10]):
    #Runs through a range of hyper parameters and returns a dataframe
    #with MSE/R2  scores for every tested combination
    etas = list()
    L2 = list()
    neurons = list()
    results_mse = list()
    results_r2 = list()
    epochs = list()
    batch_sizes = list()
    #Create time stamp to track calculation times
    t0 = time()
    #To keep track of progress
    counter = 0
    goal = len(learning_rates)*len(lambdas)*len(hidden)*len(n_epochs)*len(
            b_sizes)
    for eta in learning_rates:
        for lmd in lambdas:
            for h in hidden: 
                for sizes in b_sizes:
                    for eps in n_epochs:
                        #Initialize network
                        input_neurons = len(X[0,:])    
                        output_neurons = 1
                        net = network((input_neurons,h,output_neurons))
                        net.cost_function = "square"
                        net.fit_function = True
                        batch_size = sizes
                        #Set number of training epochs
                        number_of_epochs = eps
                        #set the training rate
                        net.learning_rate = eta
                        #Set regularization
                        net.L2 = lmd
                        counter += 1
                        #Keep track of raining progress and time usage
                        print("\rRunning time: {:.1f} Iteration: {}/{}".format(
                                time()-t0,counter,goal),end="",flush=True)
                        for e in range(number_of_epochs):
                            #Create random index vector
                            k = net.make_batch_index(len(X_train[0,:]))
                            #Calculate number of batches and loop through
                            for i in range(len(k)//batch_size):
                                batch = k[i*batch_size:(i+1)*batch_size]
                                net.train(X_train[:,batch],
                                                  y_train[batch])
                        #Calculate results for current settings
                        z = net.test(X.T,true)
                        r2 = R2(true,z[-1])
                        mse = MSE(true,z[-1])
                        results_mse.append(mse)
                        results_r2.append(r2)
                        etas.append(eta)
                        L2.append(lmd)
                        neurons.append(h)
                        epochs.append(eps)
                        batch_sizes.append(sizes)
   

    data = {"MSE":results_mse,"R2":results_r2,"Learning rate":etas,
            "L2 factor":L2,"Hidden Neurons":neurons,
            "Batch size":batch_sizes,"Number epochs":epochs}
    return pd.DataFrame(data)

def optimize_franke():
    #Calls the grid search function with  sets of hyperparameters
    #and returns a dataframe containing R2/MSE values for 
    #All combinations
    learning_rates = np.logspace(-3,0,4)
    L2 = np.logspace(-5,-1,5)
    hidden = [50,100]
    epochs =[5,10,50]
    batch_sizes = [5,10]
    hope = grid_search(learning_rates,L2,hidden,n_epochs=epochs,
                       b_sizes=batch_sizes)
    return hope

def fit_franke():
    #Uses optimized hyperparameters found with the gridsearch
    #function and tests performance against
    #validation set
    #Initialize network
    input_neurons = len(X2[0,:])    
    output_neurons = 1
    net = network((input_neurons,100,output_neurons))
    net.cost_function = "square"
    net.fit_function = True
    #Set batch size
    batch_size = 5
    #Set number of training epochs
    number_of_epochs = 500
    #set the training rate
    net.learning_rate = 1
    #Set regularization
    net.L2 = 0
    
    results_mse = list()
    results_r2 = list()
    epochs = np.arange(number_of_epochs)
    #Create time stamp to track calculation times
    t0 = time()
    for e in range(number_of_epochs):
        #Keep track of raining progress and time usage
        print("\rRunning time after {} epoch: {}".format(
                e,time()-t0),end="",flush=True)
        #Create random index vector
        k = net.make_batch_index(len(X_train[0,:]))
        #Calculate number of batches and loop through
        for i in range(len(k)//batch_size):
            batch = k[i*batch_size:(i+1)*batch_size]
            net.train(X_train[:,batch],
                              y_train[batch])
        #Calculate results for current settings
        z = net.test(X2.T,true2)
        r2 = R2(true2,z[-1])
        mse = MSE(true2,z[-1])
        results_mse.append(mse)
        results_r2.append(r2)
        
    data = {"MSE":results_mse,"R2":results_r2,"Number epochs":epochs}
    return pd.DataFrame(data)

#Setting the number of datapoints, the amount of noise 
#in the Franke function and the order of the polinomial
#used as a model.
number_of_datapoints = 100
level_of_noice = 0.1
#Making the input and output vektors of the dataset
x, y = make_data(number_of_datapoints)
z , noice = franke_function(x, y, level_of_noice)
#Flattening matrices for easier handling
xDim1 = np.ravel(x)
yDim1 = np.ravel(y)
#Frankes function withou noice
true = np.ravel(z)
#Frankes function with noice
noicy = true + np.ravel(noice)
#Create design matrix
X = design_matrix(np.ravel(x),np.ravel(y))
#Transpose data matrices to accomodate network
X_train = X.T
y_train = noicy

#Create a validation data set
x2,y2 = make_data(number_of_datapoints*2)
z , noice = franke_function(x2, y2, level_of_noice)
true2 = np.ravel(z)
X2 = design_matrix(np.ravel(x2),np.ravel(y2))

def main():
    df = fit_franke()
    ax = df.plot(x="Number epochs",y="MSE")
    fig = ax.get_figure()
    fig.savefig("Frankemse.png")
    ax2 = df.plot(x="Number epochs",y="R2")
    fig2 = ax2.get_figure()
    fig2.savefig("FrankeR2.png")
    print("MSE: ",df["MSE"].iloc[-1])
    print("MSE: ",df["R2"].iloc[-1])
if __name__ == "__main__":
    main()