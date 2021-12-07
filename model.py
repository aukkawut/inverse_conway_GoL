import numpy as np
from game import *
from pipeline import *
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, Sequential
import tensorflow as tf



def evaluation_one(grid,outcome,dt):
    '''
    This function will evaluate score for one output game at timestep t + dt. We want to minimize the score.
    '''
    score = 0
    for i in range(dt):
        outcome = one_iter(outcome)
    for i in range(len(outcome)):
        for j in range(len(outcome[0])):
            if outcome[i][j] != grid[i][j]:
                score = score + 1
    return score

def evaluation_sanity_check():
    '''
    This function will evaluate the sanity check of the evaluation function
    '''
    grid = create_grid(10)
    pos = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    grid = fill_grid(pos, grid)

    grid2 = one_iter(one_iter(grid))

    assert evaluation_one(grid2, grid,2) == True #should be True
    assert evaluation_one(grid2, grid,1) == False #should be False

def test_train(data):
    '''
    This function will split the data into training and testing data
    '''
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,1:627], data.iloc[:,626:1251], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def CNN_pipeline(data):
    '''
    This function will split data into training and testing while making it into 2 channel matrix form
    '''
    X_train, X_test, y_train, y_test = test_train(data)
    X_train = X_train[:,1:626].values.reshape(X_train.shape[0],25,25)
    X_train2 = np.full((25,25),X_train[:,:,0])
    X_test = X_test.values.reshape(X_test.shape[0],25,25)
    return X_train, X_test, y_train, y_test

def FullyConnected():
    '''
    This function will create the FullyConnected NN model
    '''
    model = Sequential()
    model.add(layers.Input((626,)))
    model.add(layers.Dense(626, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(5, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(625, activation='relu'))
    model.add(layers.Dense(625, activation='softmax'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def train_FullyConnected(x_train, y_train, batch_size = 32, epochs = 10):
    '''
    This function will train the fully connected neural network model
    '''
    model = FullyConnected()
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return model, hist

def saveModel(model,name='./model/model.h5'):
    '''
    This function will save the model
    '''
    model.save(name)

def plot4matrices(grid,grid2,grid3,grid4):
    '''
    This function will plot the 3 matrices
    '''
    fig, axs = plt.subplots(1, 4, figsize=(15,5))
    axs[0].imshow(grid, cmap='gray_r', interpolation='nearest')
    axs[1].imshow(grid2, cmap='gray_r', interpolation='nearest')
    axs[2].imshow(grid3, cmap='gray_r', interpolation='nearest')
    axs[3].imshow(grid4, cmap='gray_r', interpolation='nearest')
    axs[0].title.set_text('Input')
    axs[1].title.set_text('Prediction')
    axs[2].title.set_text('Ground Truth')
    axs[3].title.set_text('Reconstruction')
    plt.show()

def fire_not_fire(grid,threshold = 0.01):
    '''
    This function will return the matrix in which for each value that greater than threshold will yield 1, otherwise 0
    '''
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i,j] > threshold:
                grid[i,j] = 1
            else:
                grid[i,j] = 0
    return grid   

def CNN():
    '''
    This function will create the CNN model
    %TODO:
        - Change the input shape to (1,625) with another input dt
        - Test and adjust the model
    '''
    in1 = layers.Input(shape=(625,))
    in2 = layers.Input(shape=(1,))
    x = layers.Reshape((25,25,1))(in1)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.flatten(x)
    y = layers.Dense(625, activation='relu',)(in2)
    x = layers.concatenate([x,y])
    x = layers.Dense(625, activation='relu')(x)
    x = layers.Dense(625, activation='relu')(x)
    out = layers.Dense(625, activation='softmax')(x)
    model = Model(inputs=[in1,in2], outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model
def prepare_CNN(data):
    '''
    This function will prepare the data for the CNN model
    '''
    X_train, X_test, y_train, y_test = test_train(data)
    X_train = X_train[:,1:626]
    X_train2 = X_train[:,:,0]
    X_test = X_test[:,1:626]
    X_test2 = X_test[:,:,0]
    return X_train, X_test, y_train, y_test, X_train2, X_test2
def train_CNN(x_train,x_train2, y_train, batch_size = 32, epochs = 10):
    '''
    This function will train the CNN model
    '''
    model = CNN()
    hist = model.fit([x_train,x_train2], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return model, hist

def GAN():
    '''
    This function will generate the convolutional cyclic generative adversarial network (CCycleGAN) 
    %TODO:
        - Write the function
    '''
    return None