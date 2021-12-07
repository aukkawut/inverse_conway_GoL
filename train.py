from game import *
from pipeline import *
from model import *
import numpy as np

np.random.seed(0) #for consistency
data = loaddata('train.csv') #load data, please put it on the same folder

X_train, X_test, y_train, y_test = test_train(data)
print("Training Fully Connected Neural Network:")
model,hist = train_FullyConnected(X_train, y_train)
save_loss(hist, 'FullyConnected')
print(model.evaluate(X_test, y_test))

'''
train CNN
'''
print("Training CNN:")
X_train, X_test, y_train, y_test, X_train2,X_test2 = prepare_CNN(data)
model2,hist = train_CNN(X_train, X_train2, y_train)
save_loss(hist, 'CNN')
print(model2.evaluate([X_test,X_test2], y_test))
print("Done!")

grid = create_grid(25)
grid = fill_grid(random_points(25,50), grid)
grid2 = one_iter(one_iter(grid))
grid2t = grid2.reshape(1,625)
grid2t = np.insert(grid2t, 0, 2, axis=1)
grid3 = model.predict(grid2t).reshape(25,25)
#print(grid3)
grid3f = fire_not_fire(grid3)
#print(grid3f)
grid4 = one_iter(one_iter(grid3f))
print(evaluation_one(grid3f, grid2,2))
plot4matrices(grid2,grid3f,grid,grid4)