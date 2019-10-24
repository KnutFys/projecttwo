# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:59:18 2019

@author: Knut H. Engvik
"""
import numpy as np
import pandas as pd
import os
from LogisticRegessor import regressor
import sklearn.datasets
from sklearn.model_selection import train_test_split
from Network import network
#Set path for datafiles
local_path = "\\DataSets\\"
file_name = "{}{}default of credit card clients.xls".format(
        os.getcwd(),local_path)

#Prepare a dictionary to catch NaN values from dataset
nanDict = {}
#Load file into Datafrom
df_taiwan = pd.read_excel(
        file_name, header=1, skiprows=0, index_col=0, na_values=nanDict)


def normalize(x):
    return (x-np.amin(x))/(np.amax(x)-np.amin(x))

def standardize(x):
    mean = x.sum()/len(x)
    var = sum(np.square(x-mean))/(len(x)-1)
    x = (x-mean)/np.sqrt(var)
    return x

def std_norm_test():
    #Creates an array to test  normalization and standardization functions
    x = np.random.randint(0,100,100)
    print("Original array")
    print(x)
    print("Test normalization")
    print("Normalized aray")
    print(normalize(x))
    print("Test standardisation")
    print("Standardized array")
    print(standardize(x))
    print("Test dataset loaded. Printing header..")
    print(df_taiwan.head)
    
def regression_test():
    #Testing regressor on selected colums of the Taiwan banking dataset
    columns = df_taiwan.columns
    #Select number of datapoints for training
    N = 10000
    intercept = np.ones(30000)
    #Pick out testing columns
    X = np.c_[intercept,df_taiwan[[columns[4],columns[20],columns[21]]]]
    y = np.array(df_taiwan[columns[23]])    
    X = normalize(X)
    #Set first N datapoints as training data
    r = regressor(X[:N,:],y[:N])
    #Set remaining datapoints as testing data
    r.test_data = y[N:]
    r.test_X = X[N:,:]
    print("Random guess:")
    r.classify()
    r.evaluate()
    r.gradient_decent()
    print("After training guess:")
    r.classify()
    r.evaluate()

def main(): 
    #Run regression test on part of dataset
    #regression_test()
    
    #Run test of standardization and normalization
    #std_norm_test()
    
if __name__ == "__main__":
    main()