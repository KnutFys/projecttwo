# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:52:29 2019

@author: Knut H Engvik
"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from Network import network
from time import time

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
        
        #NOTE
        #As thes columns were not used, no need to remove entries
        #Remove entries with undefined marital status
        #df_taiwan = df_taiwan[df_taiwan.MARRIAGE != 0]
        #Remove entries with undefined education levels
        
        #df_taiwan = df_taiwan[df_taiwan.EDUCATION != 0]
        #df_taiwan = df_taiwan[df_taiwan.EDUCATION != 5]
        #df_taiwan = df_taiwan[df_taiwan.EDUCATION != 6]

    return df_taiwan

def normalize(x):
    return (x-np.amin(x))/(np.amax(x)-np.amin(x))


def grid_search(X_train,X_test,y_train,y_test,learning_rates,lambdas,hidden,
                n_epochs=[5],b_sizes=[10]):
    #Runs through a range of hyper parameters and returns a dataframe
    #with accuracy for every tested combination
    etas = list()
    L2 = list()
    neurons = list()
    results_accuracy = list()
    results_bal_accuracy = list()
    training_results_accuracy = list()
    training_results_bal_accuracy = list()
    epochs = list()
    batch_sizes = list()
    #Create time stamp to track calculation times
    t0 = time()
    for eta in learning_rates:
        for lmd in lambdas:
            for h in hidden: 
                for sizes in b_sizes:
                    for eps in n_epochs:
                        #Initialize network
                        input_neurons = len(X_train[:,0])  
                        print(input_neurons)
                        output_neurons = 1
                        net = network((input_neurons,h,output_neurons),
                                      smoosh_weights=False)                
                        batch_size = sizes
                        #Set number of training epochs
                        number_of_epochs = eps
                        #set the training rate
                        net.learning_rate = eta
                        #Set regularization
                        net.L2 = lmd 
                        #Set biases to 1
                        #net.set_bias()
                        print("LR: {} Hidden:{} L2:{}".format(eta,h,lmd))
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
                            #Add results for current weights, biases
                            #On testing set
                        accuracy ,confusion = net.test(X_test,y_test)
                        TP = confusion[0,0]
                        TN = confusion[1,1]
                        FP = confusion[0,1]
                        FN = confusion[1,0]
                        bal_accuracy = ((TP/(TP+FN))+(TN/(TN+FP)))/2
                        #Strore data in preparation to make datarame
                        results_accuracy.append(accuracy)
                        results_bal_accuracy.append(bal_accuracy)
                        etas.append(eta)
                        L2.append(lmd)
                        neurons.append(h)
                        epochs.append(eps)
                        batch_sizes.append(sizes)
                        #On training set
                        accuracy ,confusion = net.test(
                                X_train,y_train)
                        TP = confusion[0,0]
                        TN = confusion[1,1]
                        FP = confusion[0,1]
                        FN = confusion[1,0]
                        bal_accuracy = ((TP/(TP+FN))+(TN/(TN+FP)))/2
                        training_results_accuracy.append(accuracy)
                        training_results_bal_accuracy.append(bal_accuracy)

    data = {"ACC":results_accuracy,
            "BACC":results_bal_accuracy,
            "tACC":training_results_accuracy,
            "tBACC":training_results_bal_accuracy,
            "Learning rate":etas,
            "L2 factor":L2,
            "Hidden Neurons":neurons,
            "Batch size":batch_sizes,
            "Number epochs":epochs}
    return pd.DataFrame(data)

def optimize_taiwan(X_train,X_test,y_train,y_test):
    #Sets the parameters for grid search and returns a
    #dataframe with all hyperparameters and corresponding results
    learning_rates = [0.01,0.1,1]
    L2 = [0,0.001,0.01,1]
    hidden = [10,50,100]
    epochs =[5,10,50]
    batch_sizes = [5,10,25]
    hope = grid_search(X_train,X_test,y_train,y_test,learning_rates,
                       L2,hidden,n_epochs=epochs,b_sizes=batch_sizes)
    return hope


def split_data(df):
    #This function separates the data into categories (people who default)
    #in order to make balanced training sets. Argument is assumed to
    #be a dataframe 
    #from taiwan banking data set
    df_no_default = df[df[columns[-1]] == 0]
    df_default = df[df[columns[-1]] == 1]
    return df_no_default, df_default
    
    
def balance_data(df):
    #Split data into categories
    df0,df1 = split_data(df)    
    #Create a sample from No default category matching the size of the 
    #default category
    sample = np.random.permutation(len(df0[columns[-1]]))
    df0 = df0.iloc[sample[:len(df1[columns[-1]])]]
    return df0.append(df1)

def make_data(df, balanced=False):
    if balanced: return make_balanced_data(df)    
    intercept = np.ones(len(df[[columns[4]]]))
    X = np.c_[intercept,df[columns[5:11]]]
    #Normalise data
    for i in range(len(X[0,:-1])):
        X[:,i+1]=normalize(X[:,i+1])
    y = np.array(df[columns[23]]) 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    X_test, X_validate, y_test,y_validate = train_test_split(X_test,y_test,
                                                             test_size=0.33)
    #Transpose data matrices to accomodate network
    X_train = X_train.T
    X_test = X_test.T
    X_validate = X_validate.T
    np.reshape(y_test,(1,len(y_test)))
    np.reshape(y_train,(1,len(y_train)))
    np.reshape(y_validate,(1,len(y_validate)))
    return X,y,X_train,X_test,X_validate, y_train,y_test,y_validate

def make_balanced_data(df):
    df = balance_data(df)
    intercept = np.ones(len(df[[columns[4]]]))
    X = np.c_[intercept,df[columns[5:11]]]
    #Normalise data
    for i in range(len(X[0,:-1])):
        X[:,i+1]=normalize(X[:,i+1])
    y = np.array(df[columns[23]]) 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    X_test, X_validate, y_test,y_validate = train_test_split(X_test,y_test,
                                                            test_size=0.33)
    #Transpose data matrices to accomodate network
    X_train = X_train.T
    X_test = X_test.T
    X_validate = X_validate.T
    np.reshape(y_test,(1,len(y_test)))
    np.reshape(y_train,(1,len(y_train)))
    np.reshape(y_validate,(1,len(y_validate)))

    return X,y,X_train,X_test,X_validate, y_train,y_test,y_validate


def metrics(net,roc=False,bacc=True,):
    #Check if ROC is to be calculated
    net.metric_acc, net.metric_con = net.test(X.T,y)
    TP = net.metric_con[0,0]
    TN = net.metric_con[1,1]
    FP = net.metric_con[0,1]
    FN = net.metric_con[1,0]
    if bacc:
        net.metric_bacc = ((TP/(TP+FN))+(TN/(TN+FP)))/2
    if roc:
        #Store threshol
        hold = net.threshold
        sensitivity = list()
        specificity = list()
        fpr = list()
        precision = list()
        for threshold in np.linspace(0,1,10):
                net.threshold=threshold
                acc, con = net.test(X.T,y)
                TP = con[0,0]
                TN = con[1,1]
                FP = con[0,1]
                FN = con[1,0]
                sen = 1 if TP+FN == 0 else TP/(TP+FN)
                sensitivity.append(sen)
                spe = 1 if FP+TN == 0 else TN/(FP+TN)
                specificity.append(spe)
                pr = 1 if TP+FP == 0 else TP/(TP+FP)
                precision.append(pr)
                fpr.append(1-spe)
                
        #Reset threshold
        net.threshold = hold
        data = {"Sensitivity":sensitivity,
                "Specificity":specificity,
                "Precision":precision,
                "FPR":fpr}
        net.metric_roc = pd.DataFrame(data)         

def train_optimized(X_train,y_train):
    #Using optimized hyper parameters found, trains a network and returns it
    #Initialize network
    input_neurons = len(X_train[:,0]) 
    output_neurons = 1
    net = network((input_neurons,100,output_neurons),smoosh_weights=False)                
    batch_size = 5
    #Test if setting biases to 1 helps
    #net.set_bias()
    #Set number of training epochs
    number_of_epochs = 50
    #set the training rate
    net.learning_rate = 1
    #Set regularization
    net.L2 = 0
    t0 = time()
    plot_e = list()
    plot_acc = list()
    plot_TP = list()
    plot_bacc = list()
    for e in range(number_of_epochs):
        net.metric_acc, net.metric_con = net.test(X_test,y_test)
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
        bacc = net.metric_con[0,0]/(net.metric_con[0,0]+net.metric_con[1,0])
        bacc += net.metric_con[1,1]/(net.metric_con[1,1]+net.metric_con[0,1])
        bacc = bacc/2
        plot_e.append(e)
        plot_acc.append(net.metric_acc)
        plot_TP.append(net.metric_con[0,0])
        plot_bacc.append(bacc)
    data = {"ACC":plot_acc,
            "BACC":plot_bacc,
            "TP":plot_TP,
            "Epochs":plot_e}
    net.history = pd.DataFrame(data)
    return net    
#Load dataset. To remove datapoints containing invalid entries
#set argument clean=True
df = load_taiwan(clean=True) 
columns = df.columns
'''
#Create balanced training data
X,y,X_train,X_test,X_validate, y_train,y_test,y_validate = make_data(
        df,balanced=True)
#Train a network using balanced data
net_b = train_optimized(X_train,y_train)
metrics(net_b,roc=True)
'''
#Use unbalanced data to train a network
X,y,X_train,X_test,X_validate, y_train,y_test,y_validate  = make_data(df)

net = train_optimized(X_train,y_train)
metrics(net,roc=True)
#Plot results
'''
ax = net_b.history.plot(x="Epochs",y="ACC")
net.history.plot(ax=ax,x="Epochs",y="ACC")
ax.legend(["Balanced","Standard"])
fig = ax.get_figure()
fig.savefig("AccvEpoch.png")

ax2 = net_b.history.plot(x="Epochs",y="BACC")
net.history.plot(ax=ax2,x="Epochs",y="BACC")
ax2.legend(["Balanced","Standard"])
fig2 = ax2.get_figure()
fig2.savefig("BaccvEpoch.png")
'''
'''
#Run gridsearch
gz = optimize_taiwan(X_train,X_test,y_train,y_test)
gz.to_excel("GridSearchFullUnBalanced2.xlsx")
'''