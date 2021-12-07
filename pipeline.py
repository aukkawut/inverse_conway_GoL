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

def plot_data(data,row,start=False):
    '''
    This function will plot the data at row row from dataset, WIP
    '''
    #convert the data to numpy array with size 25,25.
    if start:
        d = data.iloc[row,2:627].to_numpy().reshape(25,25)
        dt = 0
    else:
        d = data.iloc[row,626:1251].to_numpy().reshape(25,25)
        dt = data.iloc[row,0]
    #plot the data
    fig = plt.imshow(d,cmap='gray_r',interpolation='nearest')
    plt.show()
    return fig, dt
def to_matrix(data, row, start=False):
    '''
    This function will convert the data at row row from dataset to 25x25 matrix.
    '''
    if start:
        d = data.iloc[row,2:627].to_numpy().reshape(25,25)
        dt = 0
    else:
        d = data.iloc[row,626:1251].to_numpy().reshape(25,25)
        dt = data.iloc[row,0]
    return d, dt
def export_data(data, row, start=False):
    '''
    This function will export the data at row row from dataset to data folder.
    '''
    fig,dt = plot_data(data,row,start)
    fig.axis('off')
    fig.savefig('data/'+str(row)+'_dt_'+str(dt)+'.png')

def save_loss(hist, filename, plot=True):
    '''
    This function will save the loss history to a file.
    '''
    pd.DataFrame(hist.history).to_csv(filename+'.csv')
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filename+'_acc.png')
    plt.clf()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filename+'_loss.png')