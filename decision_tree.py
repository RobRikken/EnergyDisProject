#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:40:16 2021

@author: tabearoeber
"""

### DECISION TREE 


#-------------------------------------
# import 
#-------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from IPython.display import display
import datetime
import time
import math
import warnings
warnings.filterwarnings("ignore")
import glob
from sklearn.tree import DecisionTreeRegressor


#-------------------------------------
# function def
#-------------------------------------

def read_label():
    label = {}
    for i in range(1, 7):
        hi = '/Users/tabearoeber/Library/Mobile Documents/com~apple~CloudDocs/Uni/Utrecht/Semester3/Data Science/ED Project/EnergyDisProject/data/low_freq/house_{}/labels.dat'.format(i)
        label[i] = {}
        with open(hi) as f:
            for line in f:
                splitted_line = line.split(' ')
                label[i][int(splitted_line[0])] = splitted_line[1].strip() + '_' + splitted_line[0]
    return label

def read_merge_data(house):
    path = '/Users/tabearoeber/Library/Mobile Documents/com~apple~CloudDocs/Uni/Utrecht/Semester3/Data Science/ED Project/EnergyDisProject/data/low_freq/house_{}/'.format(house)
    file = path + 'channel_1.dat'
    df = pd.read_table(file, sep = ' ', names = ['unix_time', labels[house][1]], 
                                       dtype = {'unix_time': 'int64', labels[house][1]:'float64'}) 
    
    num_apps = len(glob.glob(path + 'channel*'))
    for i in range(2, num_apps + 1):
        file = path + 'channel_{}.dat'.format(i)
        data = pd.read_table(file, sep = ' ', names = ['unix_time', labels[house][i]], 
                                       dtype = {'unix_time': 'int64', labels[house][i]:'float64'})
        df = pd.merge(df, data, how = 'inner', on = 'unix_time')
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df = df.set_index(df['timestamp'].values)
    df.drop(['unix_time','timestamp'], axis=1, inplace=True)
    return df


###------------
### PLOT DF

# plot signals of all appliances
def plot_df(df, title):
    apps = df.columns.values
    num_apps = len(apps) 
    fig, axes = plt.subplots((num_apps+1)//2,2, figsize=(24, num_apps*2) )
    for i, key in enumerate(apps):
        axes.flat[i].plot(df[key], alpha = 0.6)
        axes.flat[i].set_title(key, fontsize = '15')
    plt.suptitle(title, fontsize = '30')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
  
# plot total energy consumption per appliance
# input: df and array of house-numbers to be plot
def plot_total_consumption(df, houses):
    
    if len(houses) <= 3: 
        nrow = 1
        ncol = len(houses)
    else:
        nrow = 2
        ncol = int(len(houses)/2)
   
    # Plot total energy sonsumption of each appliance from two houses
    fig, axes = plt.subplots(nrow,ncol,figsize=(30, 10))
    plt.suptitle('Total enery consumption of each appliance', fontsize = 35)
    
    if len(houses) <= 3:
        for h in houses:
            i = list(houses).index(h)+1
            cons = df[h][df[h].columns.values[2:]].sum().sort_values(ascending=False)
            app = cons.index
            y_pos = np.arange(len(app))
            axes[i-1].bar(y_pos, cons.values,  alpha=0.6) 
            plt.sca(axes[i-1])
            plt.xticks(y_pos, app, rotation = 45)
            plt.title('House {}'.format(h), fontsize=25)

    else:
        for h in houses[:int(len(houses)/2)]:
            i = list(houses).index(h)+1
            cons = df[h][df[h].columns.values[2:]].sum().sort_values(ascending=False)
            app = cons.index
            y_pos = np.arange(len(app))
            axes[0,i-1].bar(y_pos, cons.values,  alpha=0.6) 
            plt.sca(axes[0,i-1])
            plt.xticks(y_pos, app, rotation = 45, fontsize=12)
            plt.title('House {}'.format(i), fontsize=15)
   
        for h in houses[int(len(houses)/2):]:
            i = list(houses).index(h)+1
            cons = df[h][df[h].columns.values[2:]].sum().sort_values(ascending=False)
            app = cons.index
            y_pos = np.arange(len(app))
            axes[1,i-int(len(houses)/2)-1].bar(y_pos, cons.values,  alpha=0.6) 
            plt.sca(axes[1,i-4])
            plt.xticks(y_pos, app, rotation = 45, fontsize=12)
            plt.title('House {}'.format(i), fontsize=15)
        
    
###------------
### DECISION TREE MULTIPLE APPLIANCES


# def calculate mean square error
def mse_loss(y_predict, y):
    return np.mean(np.square(y_predict - y)) 
# def calculate mean absolute error
def mae_loss(y_predict, y):
    return np.mean(np.abs(y_predict - y))


def tree_reg(appliance, training_house = 1, percentage_training_set = 0.5, plot_loss = False):
    
    # split training house in 0.5 training and 0.5 validation set
    #df1_train = df[training_house].loc[:dates[1][10]]
    #df1_val = df[training_house].loc[dates[1][11]:dates[1][16]]
    df1_train = df[training_house].loc[:dates[1][int(len(dates[1])*percentage_training_set)]]
    df1_val = df[training_house].loc[dates[1][int(len(dates[1])*percentage_training_set)]:]
    
    # for each set, determine x-values (main1 and main2)
    X_train = df1_train[['mains_1','mains_2']].values 
    X_val = df1_val[['mains_1','mains_2']].values
    
    # y-values -> dependent on appliance
    y_train = df1_train[appliance].values # get y-values from train set
    y_val = df1_val[appliance].values # get y-value from validation set
    
    min_samples_split=np.arange(2, 400, 10) # try different values for min_samples_split
    
    clfs = []
    losses = []
    start = time.time()
    # build models and calculate losses
    for split in min_samples_split: # for every split value
        clf = DecisionTreeRegressor(min_samples_split = split) # define the regression tree
        clf.fit(X_train, y_train) # fit the tree using train data
        y_predict_val = clf.predict(X_val) # extract predicted values
        clfs.append(clf)
        losses.append( mse_loss(y_predict_val, y_val) ) # calculate mse using validation set
    
    ind = np.argmin(losses) # model with smallest loss
    tree_model = clfs[ind]
    
    print('Trainning time: ', time.time() - start) # training time
    
    if plot_loss == True:
        plot_losses(losses, min_samples_split)
        
    return tree_model



def predictions(model, test_house, appliance, plot = True):
    
    # get test data from test house
    df1_test = df[test_house]
    X_test = df[test_house][['mains_2','mains_1']].values
    y_test = df[test_house][appliance].values

    y_pred = model.predict(X_test)
    
    mse_tree = mse_loss(y_pred, y_test)
    mae_tree = mae_loss(y_pred, y_test)
    
    print('Mean square error on test set: ', mse_tree)
    print('Mean absolute error on the test set: ', mae_tree)
    
    if plot == True:
        plot_each_app(df1_test, dates[test_house][12:15], y_pred, y_test, 
                      title= 'Real and predict '+ appliance + ' of house' + str(test_house))
        
    return y_pred, mse_tree, mae_tree



def plot_each_app(df, dates, predict, y_test, title, look_back = 0):
    num_date = len(dates)
    fig, axes = plt.subplots(num_date,1,figsize=(24, num_date*5) )
    plt.suptitle(title, fontsize = '25')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    for i in range(num_date):
        if i == 0: l = 0
        ind = df.loc[dates[i]].index[look_back:]
        axes.flat[i].plot(ind, y_test[l:l+len(ind)], color = 'blue', alpha = 0.6, label = 'True value')
        axes.flat[i].plot(ind, predict[l:l+len(ind)], color = 'red', alpha = 0.6, label = 'Predicted value')
        axes.flat[i].legend()
        l = len(ind)
        

def plot_losses(losses, min_samples_split):
    index = np.arange(len(min_samples_split))
    bar_width = 0.4
    opacity = 0.35

    plt.bar(index, losses, bar_width, alpha=opacity, color='b')
    plt.xlabel('min_samples_split', fontsize=30)
    plt.ylabel('loss', fontsize=30)
    plt.title('validation losses by min_samples_split', fontsize = '25')
    plt.xticks(index + bar_width/2, min_samples_split, fontsize=20 )
    plt.yticks(fontsize=20 )
    plt.rcParams["figure.figsize"] = [24,15]
    plt.tight_layout()






#-------------------------------------
# main
#-------------------------------------

if __name__ == '__main__':
# get labels for houses 1 and 2
    labels = read_label()
    for i in range(1,7):
        print('House {}: '.format(i), labels[i], '\n')
    
    # merge data from channels into one df
    df = {}
    for i in range(1,7): # for house 1 and 2
        df[i] = read_merge_data(i)
        
    # inspect df's by looking at shape and tail     
    for i in range(1,7):
        print('House {} data has shape: '.format(i), df[i].shape)
        display(df[i].tail(3))
    
    
    # extract dates from timestamps
    dates = {}
    for i in range(1,7): # for house 1 and house 2
        dates[i] = [str(time)[:10] for time in df[i].index.values]
        dates[i] = sorted(list(set(dates[i])))
        print('House {0} data contain {1} days from {2} to {3}.'.format(i,len(dates[i]),dates[i][0], dates[i][-1]))
        print(dates[i], '\n')
        
    
    ###------------
    ### VISUALISING THE DATA
    
    # Plot first/second day data of house 1 and 2
    
    for i in range(1,7):
        plot_df(df[i].loc[:dates[i][1]], 'First 2 day data of house {}'.format(i))
    
    # total energy consumption of all houses
    plot_total_consumption(df, houses=[1,2,3])    
    
    
    
    ###------------
    ### MODEL
    
    training_house = 1
    training_appliance_name = "refrigerator_5"
    
    test_house = 2
    test_appliance_name = "refrigerator_9"
    
    
    # the appliances have different names in each of the houses
    # check which appliance you'd like to predict beforehand
    
    # appliances of training house
    appliances_training_house = list(df[training_house].columns.values[2:])
    print(appliances_training_house)
    
    # appliances of test house
    appliances_test_house = list(df[test_house].columns.values[2:])
    print(appliances_test_house)
    
    
    # train the model
    tree_model = tree_reg(training_house=1, appliance = training_appliance_name, 
                          percentage_training_set=0.45, plot_loss=(True))
    
    
    # use model to predict another house
    test_house_predictions, test_house_mse, test_house_mae = predictions(tree_model, test_house, test_appliance_name, plot=False)



    
    #### DOES NOT REALLY MAKE SENSE CAUSE MODEL WAS TRAINED & VALIDATED USING COMPLETE HOUSE 1
    # test set
    # use the model to predict part of house 1 (test set)
   # df1_test = df[1].loc[dates[1][17]:] # day 17- end (day 23) for testing
   # X_test1 = df1_test[['mains_1','mains_2']].values
   # y_test1 = df1_test['refrigerator_5'].values
    
   # y_test_predict_1 = tree_model.predict(X_test1)
   # mse_tree_1 = mse_loss(y_test_predict_1, y_test1) # derive mse
   # mae_tree_1 = mae_loss(y_test_predict_1, y_test1) # derive mae
   # print('Mean square error on test set: ', mse_tree_1)
   # print('Mean absolute error on the test set: ', mae_tree_1)

    # Plot real and predict refrigerator consumption on six days of test data
   # plot_each_app(df1_test, dates[1][19:22], y_test_predict_1, y_test1, 'Real and predict Refrigerator on 3 test day of house 1')

    
    
    
    
    
    

    







