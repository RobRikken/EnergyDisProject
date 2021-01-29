#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:27:46 2021

@author: tabearoeber
"""

## supervised learning for energy disaggregation
## https://github.com/minhup/Energy-Disaggregation


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
import pathlib

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



###------------
### DECISION TREE
# def calculate mean square error
def mse_loss(y_predict, y):
    return np.mean(np.square(y_predict - y)) 
# def calculate mean absolute error
def mae_loss(y_predict, y):
    return np.mean(np.abs(y_predict - y))

# decision tree model for specific appliance
# hence, you got x_train, y_train, x_val, and y_val already
def tree_reg(X_train, y_train, X_val, y_val, min_samples_split):
    clfs = []
    losses = []
    start = time.time()
    for split in min_samples_split: # for every split value
        clf = DecisionTreeRegressor(min_samples_split = split) # define the regression tree
        clf.fit(X_train, y_train) # fit the tree using train data
        y_predict_val = clf.predict(X_val) # extract predicted values
        clfs.append(clf)
        losses.append( mse_loss(y_predict_val, y_val) ) # calculate mse using validation set
    print('Trainning time: ', time.time() - start) # training time
    return clfs, losses


# decision tree model on other appliances
def tree_reg_mult_apps():
    start = time.time()
    min_samples_split=np.arange(2, 400, 10) # try different values for min_samples_split
    pred = {}
    for app in appliances: # for each appliance
        list_clfs = []
        losses = []
        y_train = df1_train[app].values # get y-values from train set
        y_val = df1_val[app].values # get y-value from validation set
        for split in min_samples_split:
            clf = DecisionTreeRegressor(min_samples_split = split)
            clf.fit(X_train1, y_train) # train model (use same X-values for each model -> i.e. main1, main2)
            y_predict_val = clf.predict(X_val1) # predict values
            list_clfs.append(clf)
             # derive mse based on predicted and actual y-values from validation set
            losses.append( mse_loss(y_predict_val, y_val) )
        ind = np.argmin(losses) # determine model with fewest loss
        pred[app] = list_clfs[ind].predict(X_test1) # extract predicted values from selected model
    print('Trainning time: ', time.time() - start)
    return pred


def error_mul_app(mul_pred):
    mse_losses = {}
    mae_losses = {}
    for app in appliances:
        mse_losses[app] = mse_loss(mul_pred[app], df1_test[app].values)
        mae_losses[app] = mae_loss(mul_pred[app], df1_test[app].values)
    return mse_losses, mae_losses


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
        
        
###------------       
### FULLY CONNECTED NEURAL NETWORK
def build_fc_model(layers):
    fc_model = Sequential()
    for i in range(len(layers)-1):
        fc_model.add( Dense(input_dim=layers[i], units= layers[i+1]))#, W_regularizer=l2(0.1)) 
        fc_model.add( Dropout(0.5) )
        if i < (len(layers) - 2):
            fc_model.add( Activation('relu') )
    fc_model.summary()
    return fc_model


def plot_losses_fc_model(train_loss, val_loss):
    plt.rcParams["figure.figsize"] = [24,10]
    plt.title('Mean squared error of train and val set on house 1')
    plt.plot( range(len(train_loss)), train_loss, color = 'b', alpha = 0.6, label='train_loss' )
    plt.plot( range(len( val_loss )), val_loss, color = 'r', alpha = 0.6, label='val_loss' )
    plt.xlabel( 'epoch' )
    plt.ylabel( 'loss' )
    plt.legend()


#-------------------------------------
# main
#-------------------------------------

# get labels for houses 1 and 2
if __name__ == '__main__':
    labels = read_label()
    for i in range(1,3):
        print('House {}: '.format(i), labels[i], '\n')

    # merge data from channels into one df
    df = {}
    for i in range(1,3): # for house 1 and 2
        df[i] = read_merge_data(i)

    # inspect df's by looking at shape and tail
    for i in range(1,3):
        print('House {} data has shape: '.format(i), df[i].shape)
        display(df[i].tail(3))


    # extract dates from timestamps
    dates = {}
    for i in range(1,3): # for house 1 and house 2
        dates[i] = [str(time)[:10] for time in df[i].index.values]
        dates[i] = sorted(list(set(dates[i])))
        print('House {0} data contain {1} days from {2} to {3}.'.format(i,len(dates[i]),dates[i][0], dates[i][-1]))
        print(dates[i], '\n')


    ###------------
    # Plot first/second day data of house 1 and 2


    for i in range(1,3):
        plot_df(df[i].loc[:dates[i][1]], 'First 2 day data of house {}'.format(i))


    ###------------
    ### PLOT TOTAL ENERGY CONSUMPTION
    ### Plot total energy consumption of each appliance from two houses
    fig, axes = plt.subplots(1,2,figsize=(24, 10))
    plt.suptitle('Total enery consumption of each appliance', fontsize = 30)
    # for each appliance take the sum of the values and sort
    cons1 = df[1][df[1].columns.values[2:]].sum().sort_values(ascending=False)
    app1 = cons1.index
    y_pos1 = np.arange(len(app1))
    axes[0].bar(y_pos1, cons1.values,  alpha=0.6)
    plt.sca(axes[0])
    plt.xticks(y_pos1, app1, rotation = 45)
    plt.title('House 1')

    cons2 = df[2][df[2].columns.values[2:]].sum().sort_values(ascending=False)
    app2 = cons2.index
    y_pos2 = np.arange(len(app2))
    axes[1].bar(y_pos2, cons2.values, alpha=0.6)
    plt.sca(axes[1])
    plt.xticks(y_pos2, app2, rotation = 45)
    plt.title('House 2')

    ###------------
    ### DECISION TREE
    ### PREDICT REFRIGERATOR ONLY
    ## train and test on house 1 – decision tree
    # Separate house 1 data into train, validation and test data
    df1_train = df[1].loc[:dates[1][10]] # first 10 days for training
    df1_val = df[1].loc[dates[1][11]:dates[1][16]] # day 11-16 for validation
    df1_test = df[1].loc[dates[1][17]:] # day 17- end (day 23) for testing
    print('df_train.shape: ', df1_train.shape)
    print('df_val.shape: ', df1_val.shape)
    print('df_test.shape: ', df1_test.shape)

    # Using mains_1, mains_2 to PREDICT REFRIGERATOR
    # for train, validation, and test set:
    # extract main1 and main2 for as x-variable
    # extract refrigerator signal as y-variable (to be predicted)
    X_train1 = df1_train[['mains_1','mains_2']].values
    y_train1 = df1_train['refrigerator_5'].values
    X_val1 = df1_val[['mains_1','mains_2']].values
    y_val1 = df1_val['refrigerator_5'].values
    X_test1 = df1_test[['mains_1','mains_2']].values
    y_test1 = df1_test['refrigerator_5'].values
    print(X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape)


    # Using decision tree regression
    # use validation data to turn the min_samples_split parameter
    # the min_samples_split parameter determines the minimum number of samples required to split an internal node
    # a node will be split until impurity is 0, or if minimum number of samples is reached
    # default: 2 (resulting in 1 sample per leaf)

    # test different values for min_samples_split
    min_samples_split=np.arange(2, 400, 10) # values range from 2 to 392, with an interval of 10
    tree_clfs_1, tree_losses_1 = tree_reg(X_train1, y_train1, X_val1, y_val1, min_samples_split)
    plot_losses(tree_losses_1, min_samples_split) # inspect mse visually

    ### Choose the best model and predict refrigerator consumption on the test set
    ind = np.argmin(tree_losses_1) # model with smallest loss
    tree_clf_1 = tree_clfs_1[ind] # get tree with smallest loss
    # use this tree to predict y using x values from test set
    y_test_predict_1 = tree_clf_1.predict(X_test1)
    mse_tree_1 = mse_loss(y_test_predict_1, y_test1) # derive mse
    mae_tree_1 = mae_loss(y_test_predict_1, y_test1) # derive mae
    print('Mean square error on test set: ', mse_tree_1)
    print('Mean absolute error on the test set: ', mae_tree_1)


    # Plot real and predict refrigerator consumption on six days of test data
    plot_each_app(df1_test, dates[1][17:], y_test_predict_1, y_test1, 'Real and predict Refrigerator on 6 test day of house 1')


    ## use decision tree model to predict refrigerator consumption on house 2
    X_2 = df[2][['mains_2','mains_1']].values # get x-values (main)
    y_2 = df[2]['refrigerator_9'].values # get y-values (refrigerator) – to be predicted
    print(X_2.shape, y_2.shape)

    y_predict_2 = tree_clf_1.predict(X_2) # predict using decision tree model and x-values of house 2
    # derive mse and mae
    mse_tree_2 = mse_loss(y_predict_2, y_2)
    mae_tree_2 = mae_loss(y_predict_2, y_2)
    print('Mean square error on test set: ', mse_tree_2)
    print('Mean absolute error on the test set: ', mae_tree_2)

    # plot predicted on all days of house 2
    plot_each_app(df[2], dates[2], y_predict_2, y_2, 'Decision tree for refrigerator: train on house 1, predict on house 2')

    ###------------
    ### DECISION TREE
    ### PREDICT OTHER APPLIANCES

    # List of other appliances in house 1:
    appliances = list(df[1].columns.values[2:])
    appliances.pop(2)
    print(appliances)

    mul_pred = tree_reg_mult_apps()

    mul_mse_tree, mul_mae_tree = error_mul_app(mul_pred)

    for app in appliances:
        m = np.mean(df1_test[app].values)
        print('mean of {0}: {1:.2f} - mse: {2:.2f} - mae: {3:.2f}'.format(app, m ,mul_mse_tree[app], mul_mae_tree[app]))

    ## for each of the appliances
    ## plot real and predicted values for the 6 days of the test-dataset

    #for app in appliances:
    #    plot_each_app(df1_test, dates[1][17:], mul_pred[app], df1_test[app].values,
    #                  '{} - real and predict on 6 day test data of house 1'.format(app))


###-----------------------------------------------------
### FULLY CONNECTED NEURAL NETWORK

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.regularizers import l2


def fully_connected_network(X_train, Y_train, folder_name: str, appliance_name: str):
    path = 'models/fully_connected_network/' + folder_name
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    fc_model = build_fc_model([2, 256, 512, 1024, 1])

    adam = Adam(lr = 1e-5)
    fc_model.compile(loss='mean_squared_error', optimizer=adam)

    checkpointer = ModelCheckpoint(
        filepath=path + "/" + appliance_name + "_weights.{epoch:02d}-{val_loss:.2f}.hdf5",
        verbose=0,
        save_best_only=True
    )

    return fc_model.fit(X_train, Y_train,
                        batch_size=512, verbose=1, epochs=200,
                        validation_split=0.33, callbacks=[checkpointer])

if __name__ == '__main__':
    model_fc = fully_connected_network()
    pred_fc_1 = model_fc.predict(X_test1).reshape(-1)

    # print('Finish trainning. Time: ', time.time() - start)
    mse_loss_fc_1 = mse_loss(pred_fc_1, y_test1)
    mae_loss_fc_1 = mae_loss(pred_fc_1, y_test1)
    print('Mean square error on test set: ', mse_loss_fc_1)
    print('Mean absolute error on the test set: ', mae_loss_fc_1)

    # extract losses
    train_loss = model_fc.history['loss']
    val_loss = model_fc.history['val_loss']
    # plot losses
    plot_losses_fc_model(train_loss, val_loss)

    # plot predicted vs. actual signal
    plot_each_app(df1_test, dates[1][17:], pred_fc_1, y_test1,
                  'FC model: real and predict Refrigerator on 6 test day of house 1')

    ## use model to predict house 2
    y_pred_fc_2 = model_fc.predict(X_2).reshape(-1)
    mse_fc_2 = mse_loss(y_pred_fc_2, y_2)
    mae_fc_2 = mae_loss(y_pred_fc_2, y_2)
    print('Mean square error on test set: ', mse_fc_2)
    print('Mean absolute error on the test set: ', mae_fc_2)

    plot_each_app(df[2], dates[2], y_pred_fc_2, y_2, 'FC model for refrigerator: train on house 1, predict on house 2')
