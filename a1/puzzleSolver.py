'''
CSE 537 - Assign-1
Furqan Baig (109971799)
'''

import sys

BLANK = -9

class State(object):
    ''' Singleton State class to track each tile in the puzzle grid '''
    
    __instance = None
    def __new__(cls, version):
        if State.__instance is None:
            State.__instance = object.__new__(cls)
            State.__instance.version = version
            # initilize grid
            State.__instance.Grid = {}
        return State.__instance


    def __init__(self, initFileName):
        # read in initial values from input file
        # CHANGE: hardcoded for now

        self.setTileValue(0,0, 1)
        self.setTileValue(0,1, 3)
        self.setTileValue(0,2, BLANK)
        self.setTileValue(1,0, 4)
        self.setTileValue(1,1, 2)
        self.setTileValue(1,2, 5)
        self.setTileValue(2,0, 7)
        self.setTileValue(2,1, 8)
        self.setTileValue(2,2, 6)

    def setTileValue(self, i, j, val):
        if i < 0 or i > self.__instance.version or j < 0 or j > self.__instance.version:
            print ('ERROR')
        self.__instance.Grid[i, j] = val

    def getTileValue(self, i, j):
        return self.__instance.Grid[i, j]

    def printGrid(self):
        print (self.__instance.Grid)
        #for i in xrange(0, self.__instance.version):
        #    for j in xrange(0, self.__instance.version):
                
    
    def __getattr__(self, name):
        return getattr(self.__instance, name)

if __name__ == '__main__':

    if len(sys.argv) != 5:
        sys.exit(
            '''Usage: python puzzleSolver.py <#Algorithm> <N> <INPUT_FILE_PATH> <OUTPUT_FILE_PATH>
            #Algorithm:\t\t 1 for A*, 2 for Memory Bounded A*
            N:\t\t\t Puzzle - 3 for 8-tile, 4 for 15-tile
            INPUT_FILE_PATH:\t self explanatory
            OUTPUT_FILE_PATH:\t self explanatory
        ''')

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

    s = State(3)
    print('version: {}'.format(s.version))
    
    s.printGrid()
