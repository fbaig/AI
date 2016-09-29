'''
CSE 537 - Assign-1
Furqan Baig (109971799)
'''

import sys

BLANK = -9
GOAL_STATE = None

class Move:
    LEFT, RIGHT, UP, DOWN = range(4)

class State(object):
    ''' State class to track each tile in the puzzle grid '''

    def __init__(self, version, initFileName=None):
        self.version = version
        # load state info from file
        if initFileName is not None:
            import csv
            with open(initFileName, 'r') as f:
                reader = csv.reader(f)
                grid = list(reader)
            # convert all to numbers
            self.Grid = [[int(x) if x else int(BLANK) for x in inner] for inner in grid]
        # goal state
        else:
            self.Grid = [[(self.version*y)+x+1 for x in range(self.version)] for y in range(self.version)]
            self.Grid[self.version-1][self.version-1] = BLANK

        # get and store blank index
        self.bi, self.bj = self.find(BLANK)
        
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

    def move(self, direction):
        ''' 
        Move blank in given direction 
        Args:
            direction (Move): Direction to move blank
        '''

        # Out of bound movement handling
        if (direction == Move.LEFT and self.bj == 0 or 
            direction == Move.RIGHT and self.bj == self.version-1 or 
            direction == Move.UP  and self.bi == 0 or 
            direction == Move.DOWN and self.bi == self.version-1):
            raise ValueError('Invalid move: Moving out of bound')
        
        # Valid move, swap blank with value from desired direction
        print ('Moving ....')
    
    def setTileValue(self, i, j, val):
        # check for out of bound indices
        if i < 0 or i > self.version or j < 0 or j > self.version:
            print ('ERROR')
        self.Grid[i][j] = val

    def getTileValue(self, i, j):
        return self.Grid[i][j]

    def printGrid(self):
        print ('--------------------------------')
        for i in range(self.version):
            for j in range(self.version):
                # just for proper indentation/spacing
                if self.Grid[i][j] > 0 and self.Grid[i][j] < 10:
                    print (self.Grid[i][j], end='   |  ')
                else:
                    print (self.Grid[i][j], end='  |  ')
            print ('')
        print ('--------------------------------')

    def __eq__(self, state2):
        ''' Overriden equality operator to check states if they are equal based on their grid values '''
        return self.Grid == state2.Grid
    
    def __getattr__(self, name):
        return getattr(self, name)

if __name__ == '__main__':

    if len(sys.argv) != 5:
        sys.exit(
            '''Usage: python puzzleSolver.py <#Algorithm> <N> <INPUT_FILE_PATH> <OUTPUT_FILE_PATH>
            #Algorithm:\t\t 1 for A*, 2 for Memory Bounded A*
            N:\t\t\t Puzzle - 3 for 8-tile, 4 for 15-tile
            INPUT_FILE_PATH:\t self explanatory
            OUTPUT_FILE_PATH:\t self explanatory
        ''')
    ########################################## CLI argument parsing ###################################
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

    ###################################################################################################
    
    s = State(3, '3.txt')
    s.printGrid()

    global GOAL
    GOAL = State(puzzle)
    GOAL.printGrid()
    print(GOAL.Grid)

    g = State(4)


    print (s.bi)
    print (s.bj)
    
    if g == GOAL:
        print ('Goal ...')
    else:
        print ('Still working ....')

    
