import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

#Define the grid
def create_grid(n):
    '''
    This function will create the grid of size n^2 with all zero cells
    '''
    grid = np.zeros((n,n))
    return grid

def fill_grid(pos, grid):
    '''
    This function will fill the grid from position list pos with 1
    '''
    for i,j in pos:
        grid[i,j] = 1
    return grid

def count_neighbours(grid, i, j):
    '''
    This function will count the 8-neighbors on grid at position i,j
    '''
    count = 0
    for x in range(-1,2):
        for y in range(-1,2):
            if x == 0 and y == 0:
                continue
            else:
                if i+x >= 0 and i+x < len(grid) and j+y >= 0 and j+y < len(grid):
                    count += grid[i+x,j+y]
    return count    



def one_iter(grid):
    '''
    This function will perform one iteration of the game of life. This is not generalized version and not optimized
    '''
    new_grid = grid.copy()
    for i in range(len(grid)):
        for j in range(len(grid)):
            if grid[i,j] == 1:
                if count_neighbours(grid, i, j) < 2: #underpopulation
                    new_grid[i,j] = 0
                elif count_neighbours(grid, i, j) > 3: #overpopulation
                    new_grid[i,j] = 0
                elif count_neighbours(grid, i, j) == 2 or count_neighbours(grid, i, j) == 3: #stasis
                    new_grid[i,j] = 1
            elif grid[i,j] == 0:
                if count_neighbours(grid, i, j) == 3: #reproduction
                    new_grid[i,j] = 1
    return new_grid

'''
test case 1
grid = create_grid(10)
pos = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
grid = fill_grid(pos, grid)
print(grid) #should be square on the top left corner
[[1. 1. 1. 0. 0. 0. 0. 0. 0. 0.]
 [1. 1. 1. 0. 0. 0. 0. 0. 0. 0.]
 [1. 1. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

grid = one_iter(grid)
print(grid) #should be the truncated diamond square shape on top left corner
[[1. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
'''
def show_grid(grid):
    '''
    This function will show the grid
    '''
    plt.imshow(grid, cmap='Greys', interpolation='nearest')
    plt.show()
def random_points(n,m):
    '''
    This function will create a list of random points
    '''
    return [tuple(i) for i in np.random.randint(n, size=(m,2))]
'''
Test case 2
grid = create_grid(10)
pos = random_points(10,10) #should create 10 random points (x,y) where x and y are integers between 0 and 9
grid = fill_grid(pos, grid)
show_grid(grid)
'''
def export(grid, name):
    '''
    This function will export the grid to a text and a image file
    '''
    np.savetxt(name+'.txt', grid, fmt='%i')
    plt.imshow(grid, cmap='Greys', interpolation='nearest')
    plt.savefig(name+'.png')

