import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
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
            if grid[i,j] > 0:
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
'''
'''
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
def animate(n, grid,interval = 100):
    '''
    This function will animate the grid for n iterations
    '''
    fig = plt.figure()
    ax = plt.axes()
    ims = []
    for i in range(n):
        grid = one_iter(grid)
        im = plt.imshow(grid, cmap='Greys', interpolation='nearest')
        ttl = plt.text(0.5, 1.01, 'iter '+str(i), horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        
        ims.append([im,ttl])
    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=False, repeat_delay=1000)
    plt.show()
    
def show_niter(grid, n):
    '''
    This function will show the grid for n iterations, deprecated. Use animate instead
    '''
    for i in range(n):
        grid = one_iter(grid)
        plt.imshow(grid, cmap='Greys', interpolation='nearest')
        plt.title('Iter ' + str(i+1))
        plt.pause(3)
        plt.clf()
'''
Test case 3
np.random.seed(1) #for consistency
grid = create_grid(10)
pos = random_points(10,50) #should create 10 random points (x,y) where x and y are integers between 0 and 9
grid = fill_grid(pos, grid)
animate(100,grid,250) #should be empty after ~80 frames
'''

def sanity_check_game():
    '''
    This function will test the correctness of the functions
    '''
    grid = create_grid(10)
    pos = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    grid = fill_grid(pos, grid)
    assert grid[0,0] == 1
    assert grid[0,1] == 1
    assert grid[0,2] == 1
    assert grid[1,0] == 1
    assert grid[1,1] == 1
    assert grid[1,2] == 1
    assert grid[2,0] == 1
    assert grid[2,1] == 1
    assert grid[2,2] == 1
    assert count_neighbours(grid, 0, 0) == 3
    assert count_neighbours(grid, 0, 1) == 4
    assert count_neighbours(grid, 0, 2) == 3
    assert count_neighbours(grid, 1, 0) == 4
    assert count_neighbours(grid, 1, 1) == 8
    assert count_neighbours(grid, 1, 2) == 5
    assert count_neighbours(grid, 2, 0) == 3
    assert count_neighbours(grid, 2, 1) == 5
    assert count_neighbours(grid, 2, 2) == 3
    assert one_iter(grid)[0,0] == 1
    assert one_iter(grid)[0,1] == 0
    assert one_iter(grid)[0,2] == 1
    assert one_iter(grid)[1,0] == 0
    assert one_iter(grid)[1,1] == 0
    assert one_iter(grid)[1,2] == 0
    assert one_iter(grid)[2,0] == 1
    assert one_iter(grid)[2,1] == 0
    assert one_iter(grid)[2,2] == 1
'''
np.random.seed(8) #for consistency
grid = create_grid(10)
grid = fill_grid(random_points(10,25), grid)
animate(100,grid)
'''