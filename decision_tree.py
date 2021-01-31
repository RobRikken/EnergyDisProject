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
from os import walk
import re
from typing import Dict
import time
import math
import warnings
warnings.filterwarnings("ignore")
import glob
from sklearn.tree import DecisionTreeRegressor
from itertools import chain


#-------------------------------------
# function def
#-------------------------------------

def read_label():
    label = {}
    for i in range(1, 7):
        hi = 'data/low_freq/house_{}/labels.dat'.format(i)
        label[i] = {}
        with open(hi) as f:
            for line in f:
                splitted_line = line.split(' ')
                label[i][int(splitted_line[0])] = splitted_line[1].strip() + '_' + splitted_line[0]
    return label

def read_merge_data(house):
    path = 'data/low_freq/house_{}/'.format(house)
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
### LOAD (CONVERTED) FILES AND COMBINE HOUSES 


def load_data_file(path: str):
    return pd.read_csv(
        path,
        sep=' ',
        header=None
    )


def save_house_files(house: str) -> None:
    (_, __, file_names) = next(walk('data/low_freq/' + house + '/'))
    file_names.remove('labels.dat')

    for file_name in file_names:
        signal_dataframe = load_data_file('data/low_freq/' + house + '/' + file_name)
        signal_dataframe.columns = ['timestamp', 'power']
        signal_dataframe.set_index('timestamp', inplace=True)
        signal_dataframe.index = pd.to_datetime(signal_dataframe.index, unit='s')
        signal_series = pd.Series(signal_dataframe['power'], signal_dataframe.index)
        signal_series.to_pickle('data/converted/' + house + '/' + re.sub("\.dat$", '', file_name) + '.pkl')


def load_house_files() -> Dict:
    (_, house_names, _) = next(walk('data/converted/'))

    files = {}
    for house_name in house_names:
        files[house_name] = {}
        labels_file = load_data_file('data/low_freq/' + house_name + '/labels.dat')
        labels = pd.Series(labels_file[1])
        labels.index = labels_file[0]
        (_, _, file_names) = next(walk('data/converted/' + house_name + '/'))
        for file_name in file_names:
            appliance_number = file_name.split("_")[1]
            appliance_number = int(re.sub("\.pkl$", '', appliance_number))
            appliance_name = labels[appliance_number] + '__' + str(appliance_number)
            files[house_name][appliance_name] = pd.read_pickle('data/converted/' + house_name + '/' + file_name)

    return files


###------------
### LOAD (CONVERTED) FILES AND COMBINE HOUSES 
def combine_houses(houses_to_train_on=['1', '2', '3']):
    #houses_to_train_on = ['3','5','6']
    
    # Now we combine three houses, and make on training set out of them.
    house_files = load_house_files()
    # House numbers are string because later they are used as a substring to check for.
    combined_mains = {}
    dates_combined_houses = {}
    d = []
    for house in house_files:
        # Get the house number for the string and check if it is in the houses to train.
        if house.split('_')[1] in houses_to_train_on:
            # Give both of the columns a different name, so we can select later.
            house_files[house]['mains__1'].name = 'mains_1'
            house_files[house]['mains__2'].name = 'mains_2'
            # The mains are here combined into a dataframe, so it can be joined in one go to the appliance.
            combined_mains[house] = pd.concat([house_files[house]['mains__1'], house_files[house]['mains__2']],
                                                 axis=1)
            for appliance in house_files[house]:
                # Do not train on the mains, they are the X_input.
                if 'mains' in appliance:
                    continue

                # Select and appliances here, or remove to train all appliances
                if 'refrigerator' in appliance:
                    # X_train should be mains, Y_train is appliance
                    appliance_series = house_files[house][appliance]
                    combined_mains[house] = pd.merge(
                        appliance_series,
                        combined_mains[house],
                        how='inner',
                        left_index=True,
                        right_index=True
                    )

            dates_combined_houses = [str(time)[:10] for time in combined_mains[house].index.values]
            dates_combined_houses = sorted(list(set(dates_combined_houses)))
            d.append(dates_combined_houses)
    
    # get only overlapping days – too few!
    #result = set(d[0])
    #for s in d[1:]:
    #    result.intersection_update(s)

    d = list(chain.from_iterable(d))
    d = np.unique(d)
    
    first = True
    for date in d:
        for house in house_files:
            # Only run the houses that are selected to run, by checking if the number is in the list.
            if house.split('_')[1] in houses_to_train_on:
                if first:
                    houses_combined_signal = combined_mains[house].loc[str(date)]
                    first = False
                else:
                    houses_combined_signal = houses_combined_signal.append(combined_mains[house].loc[str(date)])

    #x_train = houses_combined_signal[['mains_1', 'mains_2']]
    #y_train = houses_combined_signal['power']

    combined_house_name = ''
    for number in houses_to_train_on:
        combined_house_name = combined_house_name + '_' + number
        
    return houses_combined_signal




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
def plot_total_consumption(df, houses, figsize=(30, 10)):
    
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
            plt.xticks(y_pos, app, rotation = 45, fontsize = 18)
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



# model 1, i.e. using signals from one house only (training house)
def model_1(appliance, df, percentage_training_set = 0.7, plot_loss = False,
            min_samples_split=np.arange(100, 600, 10), predict = True, plot = True):
    
    # split training house in 0.5 training and 0.5 validation set
    #df1_train = df[training_house].loc[:dates[1][10]]
    #df1_val = df[training_house].loc[dates[1][11]:dates[1][16]]
    df1_train = df[:int(len(df)*percentage_training_set)]
    df1_val = df[int(len(df)*percentage_training_set):]
    
    # for each set, determine x-values (main1 and main2)
    X_train = df1_train[['mains_1','mains_2']].values 
    X_val = df1_val[['mains_1','mains_2']].values
    
    # y-values -> dependent on appliance
    y_train = df1_train[appliance].values # get y-values from train set
    y_val = df1_val[appliance].values # get y-value from validation set
    
    # min_samples_split=np.arange(100, 600, 10) # try different values for min_samples_split
    
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
        
    y_pred_val = tree_model.predict(X_val)

    mse_tree = mse_loss(y_pred_val, y_val)
    mae_tree = mae_loss(y_pred_val, y_val)

    print('Mean square error on test set: ', mse_tree)
    print('Mean absolute error on the test set: ', mae_tree)


    if plot == True:

        dates_test = {}
        dates_test = [str(time)[:10] for time in df1_val.index.values]
        dates_test = sorted(list(set(dates_test)))

        plot_each_app(df1_val, dates_test,
                      y_pred_val, y_val, title= 'Real and predict '+ appliance + ' of house ' + str(training_house))    
    
    return tree_model


# model 2, i.e. using combined signals
def model_2(houses = ['1', '2', '3'], appliance = 'power', percentage_training_set = 0.70, 
            predict = True, plot = False):

    houses_combined_signal = combine_houses(houses_to_train_on = houses)
    
    houses_combined_train = houses_combined_signal[:int(len(houses_combined_signal)*percentage_training_set)]
    houses_combined_test = houses_combined_signal[int(len(houses_combined_signal)*percentage_training_set):]
    
    # for each set, determine x-values (main1 and main2)
    X_train = houses_combined_train[['mains_1','mains_2']].values 
    X_val = houses_combined_test[['mains_1','mains_2']].values
    
    # for each set (training and val/test) get y-values (power)
    y_train = houses_combined_train[appliance].values # get y-values from train set
    y_val = houses_combined_test[appliance].values # get y-value from validation set
    
    min_samples_split=np.arange(200, 600, 10) # try different values for min_samples_split
    
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
    
    plot_loss = True
    if plot_loss:
        plot_losses(losses, min_samples_split)
        
    if predict: 
        y_pred_val = tree_model.predict(X_val)

        mse_tree = mse_loss(y_pred_val, y_val)
        mae_tree = mae_loss(y_pred_val, y_val)
    
        print('Mean square error on test set: ', mse_tree)
        print('Mean absolute error on the test set: ', mae_tree)
        
        dates_test = {}
        dates_test = [str(time)[:10] for time in houses_combined_test.index.values]
        dates_test = sorted(list(set(dates_test)))
        
        if plot: 
            plot_each_app(houses_combined_test, dates_test, 
                      y_pred_val, y_val, title= 'Real and predict '+ appliance + ' of house ' + str(training_house))
    
    
    return tree_model



# make predictions on another house
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
        plot_each_app(df1_test, dates[test_house], y_pred, y_test, 
                      title= 'Real and predict '+ appliance + ' of house' + str(test_house))
        
    return y_pred, mse_tree, mae_tree


###------------
### TREE PLOTS


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
    plot_total_consumption(df, houses=[1,2,3], figsize = (60,12)) 
    
    
    
    ###------------
    ### MODEL
    
    print("MODEL 1")

    training_house = 1
    training_appliance_name = "refrigerator_5"
    

    # train the model
    tree_model = model_1(df = df[training_house], appliance = training_appliance_name,
                          percentage_training_set=0.70, plot_loss=(True))
    
    # use model to predict another house
    print("use model 1 on house 2")
    test_house = 2
    #print(list(df[test_house].columns.values[2:]))
    test_appliance_name = "refrigerator_9"

    test_house_predictions, test_house_mse, test_house_mae = predictions(tree_model,
                                                                         test_house,
                                                                         test_appliance_name,
                                                                         plot=True)


    print("use model 1 on house 3")
    test_house = 3
    #print(list(df[test_house].columns.values[2:]))
    test_appliance_name = "refrigerator_7"

    test_house_predictions, test_house_mse, test_house_mae = predictions(tree_model,
                                                                         test_house,
                                                                         test_appliance_name,
                                                                         plot=True)

    print("use model 1 on house 5")
    test_house = 5
    #print(list(df[test_house].columns.values[2:]))
    test_appliance_name = "refrigerator_18"

    test_house_predictions, test_house_mse, test_house_mae = predictions(tree_model,
                                                                         test_house,
                                                                         test_appliance_name,
                                                                         plot=True)


    print("use model 1 on house 6")
    test_house = 6
    #print(list(df[test_house].columns.values[2:]))
    test_appliance_name = "refrigerator_8"
    
    test_house_predictions, test_house_mse, test_house_mae = predictions(tree_model, 
                                                                         test_house, 
                                                                         test_appliance_name, 
                                                                         plot=True)

    ###------------
    ### MODEL 2
    
    print("MODEL 2")

    # aggregate houses 1,2,3
    tree_model2 = model_2(plot=True)
            
    # use model to predict another house

    print("use model 2 on house 5")
    test_house = 5
    #print(list(df[test_house].columns.values[2:]))
    test_appliance_name = "refrigerator_18"    

    test_house_predictions, test_house_mse, test_house_mae = predictions(tree_model2, 
                                                                         test_house, 
                                                                         test_appliance_name, 
                                                                         plot=True)

    print("use model 2 on house 6")
    test_house = 6
    #print(list(df[test_house].columns.values[2:]))
    test_appliance_name = "refrigerator_8"

    test_house_predictions, test_house_mse, test_house_mae = predictions(tree_model2,
                                                                         test_house,
                                                                         test_appliance_name,
                                                                         plot=True)



    # aggregate houses 3,5,6
    tree_model3 = model_2(houses=['3', '5', '6'], plot = True)
            
    # use model to predict another house
    print("use model 3 on house 1")
    test_house = 1
    #print(list(df[test_house].columns.values[2:]))
    test_appliance_name = "refrigerator_5"

    test_house_predictions, test_house_mse, test_house_mae = predictions(tree_model3,
                                                                         test_house,
                                                                         test_appliance_name,
                                                                         plot=True)


    print("use model 3 on house 2")
    test_house = 2
    #print(list(df[test_house].columns.values[2:]))
    test_appliance_name = "refrigerator_9"    

    test_house_predictions, test_house_mse, test_house_mae = predictions(tree_model3, 
                                                                         test_house, 
                                                                         test_appliance_name, 
                                                                         plot=True)
