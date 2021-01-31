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

import matplotlib.pyplot as plt
import warnings
import pathlib
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
warnings.filterwarnings("ignore")


#-------------------------------------
# function def
#-------------------------------------
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
