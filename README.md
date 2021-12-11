# Create a Time Machine for Conway’s Game of Life
This repository is the final project for DS 595 Optimization for Deep Learning and Machine Learning class at WPI on Fall 2021.

## Progress
Still WIP. Nothing functional (except for generating game)

To generate the game animation, you can create a new python file and import `game.py` file like this.

```python
from game import *
import numpy as np

np.random.seed(8) #for consistency
grid = create_grid(10) #create a 10x10 grid
grid = fill_grid(random_points(10,25), grid) #fill grid with 25 random points randomly selected from {0,1,2,...,9}^2
#we can fill in the pattern that you want in term of list of ordered pair (x,y) where (x,y) is the filled position i.e.
#pos = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)] #will create the position list that contains the points that will create the 3*3 square at the grid corner.
animate(100,grid) #animate 100 iterations of that particular orientation
```
Update: Added the `FullyConnected`, `CNN`, and `CycleGAN` (still buggy on callback routine) models.

It is not the perfect work (instead, the completely opposite), but the model be able to somewhat track where to change not what to change. If we change the architecture a little bit, it might work well.
