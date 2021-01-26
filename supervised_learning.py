from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# from IPython.display import display
import datetime
import time
import math
import warnings
import glob
from sklearn.tree import DecisionTreeRegressor


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


def read_merge_data(house: int, labels):
    path = 'data/low_freq/house_{}/'.format(house)
    file = path + 'channel_1.dat'
    df = pd.read_table(file, sep=' ', names=['unix_time', labels[house][1]],
                       dtype={'unix_time': 'int64', labels[house][1]: 'float64'})

    num_apps = len(glob.glob(path + 'channel*'))
    for i in range(2, num_apps + 1):
        file = path + 'channel_{}.dat'.format(i)
        data = pd.read_table(file, sep=' ', names=['unix_time', labels[house][i]],
                             dtype={'unix_time': 'int64', labels[house][i]: 'float64'})
        df = pd.merge(df, data, how='inner', on='unix_time')
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df = df.set_index(df['timestamp'].values)
    df.drop(['unix_time', 'timestamp'], axis=1, inplace=True)

    return df


#def decision_tree_regression():
    # Separate house 1 data into train, validation and test data
    # df1_train = df[1].ix[:dates[1][10]]
    # df1_val = df[1].ix[dates[1][11]:dates[1][16]]
    # df1_test = df[1].ix[dates[1][17]:]
    #
    # X_train1 = df1_train[['mains_1', 'mains_2']].values
    # y_train1 = df1_train['refrigerator_5'].values
    # X_val1 = df1_val[['mains_1', 'mains_2']].values
    # y_val1 = df1_val['refrigerator_5'].values
    # X_test1 = df1_test[['mains_1', 'mains_2']].values
    # y_test1 = df1_test['refrigerator_5'].values

def mse_loss(y_predict, y):
    return np.mean(np.square(y_predict - y))
def mae_loss(y_predict, y):
    return np.mean(np.abs(y_predict - y))

def tree_reg(X_train, y_train, X_val, y_val, min_samples_split):
    clfs = []
    losses = []
    start = time.time()
    for split in min_samples_split:
        clf = DecisionTreeRegressor(min_samples_split = split)
        clf.fit(X_train, y_train)
        y_predict_val = clf.predict(X_val)
        clfs.append(clf)
        losses.append( mse_loss(y_predict_val, y_val) )
    print('Trainning time: ', time.time() - start)
    return clfs, losses

# using decision tree model on other appliances
def tree_reg_mult_apps(appliances, df1_train, df1_val, X_train1, X_val1, X_test1):
    min_samples_split = np.arange(2, 400, 10)
    pred = {}
    for app in appliances:
        list_clfs = []
        losses = []
        y_train = df1_train[app].values
        y_val = df1_val[app].values
        for split in min_samples_split:
            clf = DecisionTreeRegressor(min_samples_split=split)
            clf.fit(X_train1, y_train)
            y_predict_val = clf.predict(X_val1)
            list_clfs.append(clf)
            losses.append(mse_loss(y_predict_val, y_val))
        ind = np.argmin(losses)
        pred[app] = list_clfs[ind].predict(X_test1)

    return pred

def error_mul_app(mul_pred, appliances, df1_test):
    mse_losses = {}
    mae_losses = {}
    for app in appliances:
        mse_losses[app] = mse_loss(mul_pred[app], df1_test[app].values)
        mae_losses[app] = mae_loss(mul_pred[app], df1_test[app].values)
    return mse_losses, mae_losses

def run_supervised_learning():
    labels = read_label()

    df = {}
    for i in range(1, 3):
        df[i] = read_merge_data(i, labels)

    # Choose the best model and predict refrigerator consumption on the test set
    # ind = np.argmin(tree_losses_1)
    # tree_clf_1 = tree_clfs_1[ind]
    # y_test_predict_1 = tree_clf_1.predict(X_test1)
    # mse_tree_1 = mse_loss(y_test_predict_1, y_test1)
    # mae_tree_1 = mae_loss(y_test_predict_1, y_test1)

    # Using decision tree model we have just trained on house 1 to predict refrigerator consumtion on house 2
    # X_2 = df[2][['mains_2','mains_1']].values
    # y_2 = df[2]['refrigerator_9'].values
    # y_predict_2 = tree_clf_1.predict(X_2)
    # mse_tree_2 = mse_loss(y_predict_2, y_2)
    # mae_tree_2 = mae_loss(y_predict_2, y_2)
    #
    # mul_pred = tree_reg_mult_apps()
    #
    # mul_mse_tree, mul_mae_tree = error_mul_app(mul_pred)