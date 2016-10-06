'''
CSE 537 - Assign-1
Furqan Baig (109971799)
'''

import sys

BLANK = -9
GOAL_STATE = None

from enum import Enum
Move = Enum('Move', 'LEFT RIGHT UP DOWN')

class State(object):
    ''' State class to track each tile in the puzzle grid '''

    def __init__(self, version, initFileName=None, grid=None, parent=None, direction=None):
        '''
        Args:
            version (int):      3 for 8 tile, 4 for 15 tile
            initFileName (str): Input file to read initial state data from
            grid (int[][]):     Initialize state from given grid state
            parent (State):     For path backtracking in algorithms
            direction (Move):   The move from parent to this state
        '''
        self.version = version
        # for backtracking
        self.parent = parent
        self.direction = direction
        # load state info from file
        if initFileName is not None:
            import csv
            with open(initFileName, 'r') as f:
                reader = csv.reader(f)
                grid = list(reader)
            # convert all to numbers
            self.Grid = [[int(x) if x else int(BLANK) for x in inner] for inner in grid]
        # create object from given state
        elif(grid is not None):
            self.Grid = grid
        # goal state
        else:
            self.Grid = [[(self.version*y)+x+1 for x in range(self.version)] for y in range(self.version)]
            self.Grid[self.version-1][self.version-1] = BLANK

        # get and store blank index
        self.bi, self.bj = self.find(BLANK)

        # the actual physical cost to reach this state from start
        self.g_cost = 0
        # heuristic cost to reach goal from this state
        self.h_cost = 0
        
    def find(self, val):
        ''' 
        Find the 2d index of given value in the Grid 
        Args:
            val (int): int value to find in the Grid
        Returns:
            (i, j): if exists, 2D index of given value in Grid
                    None otherwise
        '''
        for i, x in enumerate(self.Grid):
            if val in x:
                return (i, x.index(val))
        return None

    def distance(self):
        ''' A simple heuristic function based on the number of misplaced tiles '''
        for i in range(self.version):
            for j in range(self.version):
                # ignore blank (last index)
                if i==self.version-1 and j==self.version-1:
                    continue
                elif self.version*i+j+1 != self.Grid[i][j]:
                    self.h_cost += 1

    def manhatan_distance(self):
        ''' Somewhat better heuristic than simple distance '''
        for i in range(self.version):
            for j in range(self.version):
                if self.Grid[i][j] != -9:
                    goal_i = int((self.Grid[i][j]-1)/self.version)
                    goal_j = int((self.Grid[i][j]-1)%self.version)
                    # horizontal distance to goal 
                    dx = i - goal_i
                    # vertical distance to goal
                    dy = j - goal_j
                    self.h_cost += abs(dx) + abs(dy)
        #self.h_cost = self.h_cost * 2
        
    def calculate_h(self):
        self.manhatan_distance()
    
    def move(self, direction):
        ''' 
        Move blank in given direction 
        Args:
            direction (Move):    Direction to move blank
        Return:
            New object after move. Note, current state remains unchanged
        '''

        # Out of bound movement handling
        if (direction == Move.LEFT and self.bj == 0 or 
            direction == Move.RIGHT and self.bj == self.version-1 or 
            direction == Move.UP  and self.bi == 0 or 
            direction == Move.DOWN and self.bi == self.version-1):
            return None
        
        # Valid move, swap blank with value from desired direction
        if (direction == Move.LEFT):
            move_index_i, move_index_j = self.bi, (self.bj-1)
        elif (direction == Move.RIGHT):
            move_index_i, move_index_j = self.bi, (self.bj+1)
        elif (direction == Move.UP):
            move_index_i, move_index_j = (self.bi-1), self.bj
        elif (direction == Move.DOWN):
            move_index_i, move_index_j = (self.bi+1), self.bj

        move_index_value = self.Grid[move_index_i][move_index_j]

        # make a copy of
        grid = [x[:] for x in self.Grid]
        
        # swap values
        grid[move_index_i][move_index_j] = BLANK
        grid[self.bi][self.bj] = move_index_value
        # update blank index
        #grid.bi, grid.bj = move_index_i, move_index_j

        return State(self.version, grid=grid, parent=self, direction=direction)
        
    def __repr__(self):
        ret = '\n------------------------\n'
        for i in range(self.version):
            for j in range(self.version):
                # just for proper indentation/spacing
                if self.Grid[i][j] > 0 and self.Grid[i][j] < 10:
                    ret += str(self.Grid[i][j]) + '   |  '
                else:
                    ret += str(self.Grid[i][j]) + '  |  '
            ret += '\n'
        ret += '------------------------\n'
        return ret

    def __lt__(self, state2):
        return (self.g_cost + self.h_cost) < (state2.g_cost + state2.h_cost)
        
    def __eq__(self, state2):
        ''' Overriden equality operator to check states if they are equal based on their grid values '''
        return self.Grid == state2.Grid

############################################## Algo ##########################################

class Algo():

    def __init__(self, start):
        self.start = start
        self.start.g_cost = 0
        self.states_count = 0
        
        
    def track_path(self, goal):
        path = []
        node = goal
        # print path
        while node.parent is not None:
            path.append(node)
            node = node.parent

        path.reverse()
        for s in path:
            print (str(s.direction) + ', ')

        print('Total states explored: ' + str(self.states_count))
        
    def apply_action(self, current, move):
        new = current.move(move)
        if new and new not in self.explored:
            self.states_count += 1
            new.g_cost = current.g_cost + 1
            # calculate h cost from heuristic function
            new.calculate_h()
            if new == GOAL:
                self.track_path(new)
                return True
            self.frontier.put(new)
            
    def bfs(self):
        ''' Simple blind Breath First Search '''
        import queue as Q

        self.frontier = Q.PriorityQueue()
        self.frontier.put(self.start)
    
        self.explored = []
        while not self.frontier.empty():
            current = self.frontier.get()
            if current == GOAL:
                print ('GOAL REACHED ...')
                return True
            self.explored.append(current)
            # generate all possible states by moving blank L, R, U, D
            if self.apply_action(current, Move.LEFT): return True
            elif self.apply_action(current, Move.RIGHT): return True
            elif self.apply_action(current, Move.UP): return True
            elif self.apply_action(current, Move.DOWN): return True
        
    def printFrontier(self):
        while not self.frontier.empty():
            print(self.frontier.get())
        
if __name__ == '__main__':

    if len(sys.argv) != 5:
        sys.exit(
            '''Usage: python puzzleSolver.py <#Algorithm> <N> <INPUT_FILE_PATH> <OUTPUT_FILE_PATH>
            #Algorithm:\t\t 1 for A*, 2 for Memory Bounded A*
            N:\t\t\t Puzzle - 3 for 8-tile, 4 for 15-tile
            INPUT_FILE_PATH:\t self explanatory
            OUTPUT_FILE_PATH:\t self explanatory
        ''')
    #################################### CLI argument parsing #############################
    algo = int(sys.argv[1])
    puzzle = int(sys.argv[2])
    inputPath = sys.argv[3]
    outputPath = sys.argv[4]

    if (algo == 1):
        print ('Algorithm: Simple A*')
    elif (algo == 2):
        print ('Algorithm: Memory bounded A*')
    else:
        sys.exit('Invalid algorithm')

    if (puzzle == 3):
        print ('Puzzle: 8-tile')
    elif (puzzle == 4):
        print ('Puzzle: 15-tile')
    else:
        sys.exit('Invalid puzzle')
        
    print ('InputFilePath: {}'.format(inputPath))
    print ('OutputFilePath: {}'.format(outputPath))

    #######################################################################################
    
    s = State(puzzle, '3.txt')
    print('Start State: ' + str(s))
    
    global GOAL
    GOAL = State(puzzle)

    GOAL.calculate_h()
    
    a = Algo(s)
    a.bfs()

    
