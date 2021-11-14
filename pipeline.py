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

def plot_data(data,row):
    '''
    This function will plot the data at row row from dataset 
    '''
    