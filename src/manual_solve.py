#!/usr/bin/python

import os
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

"""
Name: Rohit Kumar Tilwani
Student ID: 21235507

The approach and required transformations are discussed for every task.

Summary: Many reasoning tasks in the corpus require identifying shapes and patterns in the form of figures
and since humans are good at finding figures and patterns in visual data, the logic of patterns is easy to spot for humans
but making a machine understand such tasks is comparatively much more difficult. With limited number of data points, rule based
approach is followed with sets of defined rules and each logic has to be tailored for a specific problem or for a specific 
domain of tasks. Moreover, many tasks that look trivial and can be usually figured out by us within a couple of seconds, are much
more compute intensive when it comes to a system doing the same.

The solve functions for all the tasks solved below use standard python libraries (numpy, itertools, collections) and standard Python
code and data structures. Also, recursion is used as a common approach to reach adjoining cells in different directions to define a 
shape in terms of component coordinates. However, as mentioned above each task has different requirement to interpret a figure or shape, 
different methods are used, suitable for that task's purpos. For instance, extracting boundaries or extent of an asymmetric figure requires 
modified approach than extracting extent of a rectangle, which can be done efficiently extracted using just the length and breadth,
without traversing all the points within it.

The flexibility Python offers in the form of weakly typed language, for instance no fixed return type of a method or the flexibility 
to store any kind of data in datastructures, helps to handle multiple scenarios within a single variable, reducing the lines of code and 
improving readability.
"""
# importing the required libraries
import itertools
from collections import defaultdict



#***************** solving task 6a1e5592 *****************************

"""
Required Transformations: This task requires to identify and keep track of the different shapes present 
at the bottom and its subset figure present at the top of the grid. Once identified, the bottom shapes must be placed 
in their correct position (at the top) after matching, which requires cross matching to find the best 
match.

For this task, various helper functions are defined which are called from solve_6a1e5592, to improve 
the readability of the code and to keep it modular. The shapes are identified in a bottom up approach
using a single point and considering adjoining points part of a single shape. The matching is done using
the top-leftmost points of both the figures and check for best matching as in some cases multiple figures
can match with a single candidate. For e.g. a base figure rectangle of 3x2 will fit in square of 2x2 
and line of 1x2 present at the top.

All the train and test grids, covering all the cases are solved
"""

# function to find shapes by traversing in three different directions (bottom-up)
"""
x and y: array indices using which adjoining points are traversed
arr: numpy array 
s: set of traversed points
n: value to be used to consider a point in the shape
"""
def find_shape(x, y, arr, s, n):
    #  checking for out of bounds, non-relevant value, and if the point is already is set
    if x < 0 or x >= arr.shape[0] or y < 0 or y >= arr.shape[1] or arr[x][y] != n or (x, y) in s: return s
    s.add((x, y))    # adding the current point to set of traversed points
    # updating the set by traversing other directions
    s = find_shape(x, y + 1, arr, s, n)  
    s = find_shape(x - 1, y, arr, s, n)
    s = find_shape(x, y - 1, arr, s, n)
    return s    # returning set containing all figure points

# function to check if two figures match each other
"""
fig_1 and fig_2: set of points of figures to be matched (figure on top and on the base respectively)
d = tuple of distances between top-leftmost points of both the figures in both the dimensions
arr = numpy array (task matrix)
"""
def match(fig_1, fig_2, d, arr):
    for point in fig_2:
        corr_point = (point[0] - d[0], point[1] - d[1])   # corresponding point in figure 1 
        try:
            if corr_point in fig_1:    # if corr_point is present in figure 1
                fig_1.remove(corr_point)
                continue
            elif arr[corr_point[0], corr_point[1]] != 0:  # if corr_point doesn't exist in top figure and is not black
                return False
        except IndexError:   # checking for indexerror (arising due to distance subtractions),
                             # arises when there is no corr_point in the matrix
            return False
    """
    # even if the base figure points match the head of the top figure, 
    # it should not leave any vacant points 
    # (e.g. a vertical line of length 2 will fit in a square of length 2 but is not a correct match)
    # since fig_1(top figure) points are a subset of base figure, they should be exhausted
    # although even this doesn't guarantee a perfect match. for e.g.,
    # a base figure rectangle of 3x2 will fit in square of 2x2 and line of 1x2, exhausting the points
    # this condition of exhausting most points is checked before filling the array, if a figure has multiple matches
    """
    if len(fig_1) != 0:
        return False
    return True

# function to return the aggregated list of figure sets in row r and columns indices of array arr
# wrt value n
def get_figures(r, indices, arr, n):
    figures = []    # variable to store sets of points of figures
    for i in range(len(indices)):
        # if two indices are contiguous, they will be part of one shape and can be skipped
        if i == 0 or indices[i - 1] != indices[i] - 1:
            # adding the black color identified shape 
            figures.append(find_shape(r, indices[i], arr, set(), n))
    return figures


def solve_6a1e5592(arr):
    matches = defaultdict(list)
    x = arr.copy()
    # setting the row values to -1 where whole row is zero (helpful to identify shapes)
    for i in range(len(x)):
        if len(np.where(x[i] != 0)[0]) == 0:
            x[i] = np.where(x[i] == 0, -1, x[i])
    
    # row to look in for top shapes
    r_top = np.where(x == -1)[0].min() - 1 
    indices = np.where(x[r_top]==0)[0]
    top_figures = get_figures(r_top, indices, x, 0)
    
    # row to look in for bottom shapes
    r_bot = x.shape[0] - 1
    indices = np.where(x[r_bot] == 5)[0]
    bot_figures = get_figures(r_bot, indices, x, 5)
    x = np.where(x == -1, 0, x)
    x = np.where(x == 5, 0, x)
    
    # matching the figures
    for f_top, f_bot in itertools.product(top_figures, bot_figures):
        # getting the top-leftmost points of top figure f_top and bottom figure f_bot
        f1 = min(f_top, key = lambda x: (x[0], x))
        f2 = min(f_bot, key = lambda x: (x[0], x))
        d = (f2[0] - f1[0], f2[1] - f1[1])   # distance in both the dimensions

        if match(f_top.copy(), f_bot.copy(), d, x):
            matches[tuple(f_bot)].append((d, f_top))  # matches for the bottom figure
    
    # filling the values for the matched figures
    for f_bot, val in matches.items():
        d, f_top = val[0]   # distance and matching top figure
        # if multiple top figures match, take the one which has more points
        # i.e. more points of the base figure fitted in the top figure frame
        if len(val) != 1:
            d, f_top = max(val, key = lambda x: len(x[1]))
        # filling the values
        for i, j in f_bot:
            x[i-d[0], j-d[1]] = 1
    return x


#***************** solving task 00d62c1b *****************************

"""
Required Transformations: Here, the major transformation is to identify the bounded shape that can occur in any shape.
This requires to cover a point in all directions since the boundary can be far away or can break in
between in any other direction.

For this, array is iterated and points of interest are checked for boundaries. The task of boundary
identification includes many overlapping subtasks, and solving them over and over can take time. To avoid
this, memory/history of previously seen points is used and adjacent are recursively called to check
for boundaries.

The approach implementented below, successfully solves all the train and test grids for the task.
"""

# helper functions to return coordinates in a particular direction wrt input point
def up(x, y): return x-1, y
def left(x, y): return x, y-1
def right(x, y): return x, y+1
def down(x, y): return x+1, y

# function to check if a point is bounded in all directions
# using recursion and reaching other points
"""
 x and y: rows and column index of the point to be checked
 arr = 2D numpy array
 memory = keeping record of points what points are identified as bounded already,
 since the problem contains overlapping subtasks, memory can save time 
"""
def bounded(x, y, arr, memory):
    # checking if the coordinate is within bounds 
    if x < 0 or x >= arr.shape[0] or y < 0 or y >= arr.shape[1]: return False
    try:
        # checking if the current point touches the wall(boundary) or is already in memory
        if arr[x][y] == 3 or arr[x][y] == 4 or memory[(x,y)] == 1: return True
        elif memory[(x,y)] == 0: return False # if a point is already in memory as unbounded
        elif memory[(x,y)] == -1: return -1   # if the current point is not yet identified as bounded/unbounded
    except: memory[(x,y)] = -1   # setting the value and checking for the current point
    
    """
    neg_flag is used as an indicator to ensure if a point is identified as bounded/unbounded
    i.e. if a point has boundary on some sides and -1 (unidentified) points on other side
    """
    neg_flag = 0
    # checking for boundaries in all four directions
    for f in [left, up, right, down]:
        val = bounded(*f(x, y), arr, memory)
        if val == False:   # if point is definitely unbounded in a direction
            memory[(x,y)] = 0
            return False
        if val == -1:      # if the boundary in a particular direction can't be confirmed, due to uncertainty of other points
            neg_flag = 1
            continue
    
    if neg_flag == 0: memory[(x,y)] = 1    # if neg_flag = 0 and there is certainty of boundaries in all directions
    else: memory.pop((x,y))        # point not added in memory in case there is not surity
    return True

def solve_00d62c1b(arr):
    memory = dict()    # initialising the memory
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if arr[x][y] == 0 and bounded(x, y, arr, memory) == True:
                arr[x][y] = 4    # setting the color code if bounded returns true for a black color point
    return arr


#***************** solving task 8efcae92 *****************************
"""
Required Transformations: The following task 8efcae92 requires to identify the figure with the maximum
number of red dots inside. The main part of this task is to correctly and efficiently identify all 
the rectangles present in the given array.

In the solve method for this task, we make use of the information that the figures are rectangles. While iterating, if
any point of a figure is reached, the length and height of the rectangle is calculated by consecutively
and incrementally checking for points in both the dimensions (which doesn't require to traverse the entire
points to confirm if they are in a figure). The red points within this rectangular space can be calculated

The above mentioned method solves all the cases (training and testing).
"""
# function to get dimension length using x and y coordinates of an array 
# using increments accross both the dimensions
def dimension_len(x, inc_x, y, inc_y, arr):
    d_len = 0    # initialising the dimension length to zero
    try:
        while arr[x][y] == 1 or arr[x][y] == 2:   # looking for length (number of columns)
            d_len += 1
            x += inc_x
            y += inc_y
    except IndexError:
        pass       # check for out of bounds
    return d_len 


# functon to identify the figures by going to adjoining points
"""
x and y = top-left point of the rectangle shape
arr = numpy array (matrix of the task)
"""
def get_fig(x, y, arr):
    # looking for length (number of columns)
    length = dimension_len(x, 0, y, 1, arr)
    height = dimension_len(x, 1, y, 0, arr)
    # returning the figure array
    return arr[x : x + height, y : y + length]

def solve_8efcae92(arr):
    max_red_dots = 0     # variable to store the max_red_dots seen so far
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if arr[x][y] == 1 or arr[x][y] == 2:   # if the current point is top-left corner of a figure
                # getting the figure array and comparing it with the max_red_dots
                fig = get_fig(x, y, arr)
                red_dots = len(np.where(fig == 2)[0])
                if red_dots > max_red_dots:
                    max_red_dots = red_dots
                    res = fig.copy()
    # res stores the figure having most red dots
    return res


#***************** solving task 05f2a901 *****************************
"""
Required Transformations: This task requires us to shift the image either across the rows or columns in
a way so that it touches the given square figure. The identification of positions of images with respect to each
other and determining the distance along with the direction to shift is required.

To identify if the given images are aligned horizontally or vertically wrt each other, overlapping of 
rows and columns is checked. Based on this alignment, the distance to shift is calculated by taking the difference
of walls of figures facing each other.

All the training and testing cases are successfully solved by the above approach.
"""

def solve_05f2a901(arr):
    r, c = np.where(arr == 2)    # figure coordinates to be moved
    r_fix, c_fix = np.where(arr == 8)   # fixed figure coordinates
    
    # if the rows of the figures are overlapping
    if r_fix.min() in r or r.min() in r_fix:
        # identifying the position of one figure wrt other (right or left)
        if abs(c_fix.min() - c.max()) < abs(c_fix.max() - c.min()):
            cols_to_move = c_fix.min() - c.max() - 1   # columns to be shifted
        else:
            cols_to_move = c_fix.max() - c.min() + 1
        arr = np.where(arr == 2, 0, arr)
        # shifting the columns
        for x, y in zip(r, c):
            arr[x][y + cols_to_move] = 2
    
    # if the columns of the figures are overlapping
    elif c_fix.min() in c or c.min() in c_fix:
        # identifying the position of one figure wrt other (top or bottom)
        if abs(r_fix.min() - r.max()) < abs(r_fix.max() - r.min()):
            rows_to_move = r_fix.min() - r.max() - 1  # rows to be shifted
        else:
            rows_to_move = r_fix.max() - r.min() + 1
        arr = np.where(arr == 2, 0, arr)
        # shifting the rows 
        for x, y in zip(r, c):
            arr[x + rows_to_move][y] = 2
    return arr


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

