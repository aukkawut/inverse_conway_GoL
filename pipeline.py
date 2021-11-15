from game import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def loaddata(filename):
    '''
    This function will read the data from the file
    '''
    data = pd.read_csv(filename)
    return data

def plot_data(data,row,end=False):
    '''
    This function will plot the data at row row from dataset, WIP
    '''
    #convert the data to numpy array with size 25,25.
    if end:
        d = data.iloc[row,1:625].to_numpy().reshape(25,25)
    else:
        d = data.iloc[row,626:1251].to_numpy().reshape(25,25)
    #plot the data
    plt.imshow(d,cmap='gray',interpolation='nearest')
    plt.show()

