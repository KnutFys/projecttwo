# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:16:42 2019

@author: Knut H Engvik
"""

import numpy as np
import pandas as pd
import os
from LogisticRegessor import regressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Set path for datafiles
local_path = "\\DataSets\\"
file_name = "{}{}default of credit card clients.xls".format(
        os.getcwd(),local_path)




def load_taiwan(clean=False):
    #Loads taiwan banking dataset. If clean is set to True, datapoints
    #with undefined values are removed
    #Prepare a dictionary to catch NaN values from dataset
    nanDict = {}
    #Load file into Datafrom
    df_taiwan = pd.read_excel(
        file_name, header=1, skiprows=0, index_col=0, na_values=nanDict)

    #Store the column names
    columns = df_taiwan.columns
    if clean:
        #Remove -2 entries from pyment history columns
        for i in range(6):
            df_taiwan = df_taiwan[df_taiwan[columns[5+i]] != -2]
        
        #Remove entries with undefined marital status
        df_taiwan = df_taiwan[df_taiwan.MARRIAGE != 0]
        #Remove entries with undefined education levels
        df_taiwan = df_taiwan[df_taiwan.EDUCATION != 0]
        df_taiwan = df_taiwan[df_taiwan.EDUCATION != 5]
        df_taiwan = df_taiwan[df_taiwan.EDUCATION != 6]

    return df_taiwan

def normalize(x):
    return (x-np.amin(x))/(np.amax(x)-np.amin(x))

#Load dataset. To remove datapoints containing invalid entries
#set argument clean=True
df = load_taiwan(clean=True) 
#Testing regressor on selected colums of the Taiwan banking dataset
columns = df.columns
intercept = np.ones(len(df[[columns[4]]]))
#Pick out testing columns and add intercept
X = np.c_[intercept,df[columns[5:11]]]
for i in range(len(X[0,:-1])):
    X[:,i+1]=normalize(X[:,i+1])
y = np.array(df[columns[23]]) 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
X_test, X_validate, y_test,y_validate = train_test_split(X_test,y_test,
                                                         test_size=0.33)
#Set first N datapoints as training data
r = regressor(X_train,y_train)
#Set remaining datapoints as testing data
r.test_data = y_test
r.test_X = X_test
#Setting hyper parameters to run through
threshold = [0,0.1,0.2,0.3,0.4,0.5,0.7,1]

def find_learningrate():
    #Runs through different learning rates and plots accuracy as
    #a function of number of gradient steps for each.
    r.threshold = 0.5
    r.iterations = 1
    r.L2 = 0.01
    eta = np.logspace(-3,0,5)
    plots = list()
    number_of_points = 1000
    for e in eta:
        r.reinitiate_weights()
        r.eta = e
        accuracy = list()
        r.classify()
        r.evaluate(verbose=False)
        tmp = r.accuracy
        accuracy.append(tmp)
        for n in range(number_of_points):
            r.gradient_decent()
            r.classify()
            r.evaluate(verbose=False)
            tmp = r.accuracy
            accuracy.append(tmp)
        plots.append((np.arange(number_of_points+1),accuracy))
    
    for index,pl in enumerate(plots):
        plt.semilogx(pl[0],pl[1],label="{:3.2e}".format(eta[index]))
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig("LearningratesL2{}Fe.png".format(r.L2),
                bbox_inches = "tight")

def find_L2():
    #Runs through different ridge penalty factors and plots accuracy as
    #a function of number of gradient steps for each.
    lmda = np.logspace(-4,1,5)
    r.threshold = 0.5
    r.iterations = 1
    r.eta = 1
    plots = list()
    number_of_points = 1000
    for lm in lmda:
        r.reinitiate_weights()
        r.L2 = lm
        accuracy = list()
        r.classify()
        r.evaluate(verbose=False)
        tmp = r.accuracy
        accuracy.append(tmp)
        for n in range(number_of_points):
            r.gradient_decent()
            r.classify()
            r.evaluate(verbose=False)
            tmp = r.accuracy
            accuracy.append(tmp)
        plots.append((np.arange(number_of_points+1),accuracy))
    
    for index,pl in enumerate(plots):
        plt.semilogx(pl[0],pl[1],label="{:3.2e}".format(lmda[index]))
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig("RegularizationLR{}.png".format(r.eta),bbox_inches = "tight")

def optimal(eta=1,L2=0):
    r.reinitiate_weights()
    r.eta = eta
    r.L2 = L2
    r.iterations = 1000
    r.gradient_decent()
    r.test_data = y_validate
    r.test_X = X_validate
    r.classify()
    r.evaluate(verbose=True)

    
def main():
    #find_L2()
    optimal()

if __name__ == "__main__":
    main()