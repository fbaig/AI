'''
CSE 537 - Assign-1
Furqan Baig (109971799)
'''

import sys

class State:
    ''' State class to track each tile in the puzzle grid '''
    
    

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
