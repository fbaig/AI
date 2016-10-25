'''
CSE 537 - Assign-1
Furqan Baig (109971799)
'''

import sys

BLANK = -9
GOAL_STATE = None
MAX = float('inf')

from enum import Enum
Move = Enum('Move', 'L R U D')

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
        self.h_cost = self.h_cost
        
    def calculate_h(self):
        #self.manhatan_distance()
        self.distance()
    
    def move(self, direction):
        ''' 
        Move blank in given direction 
        Args:
            direction (Move):    Direction to move blank
        Return:
            New object after move. Note, current state remains unchanged
        '''

        # Out of bound movement handling
        if (direction == Move.L and self.bj == 0 or 
            direction == Move.R and self.bj == self.version-1 or 
            direction == Move.U  and self.bi == 0 or 
            direction == Move.D and self.bi == self.version-1):
            return None
        
        # Valid move, swap blank with value from desired direction
        if (direction == Move.L):
            move_index_i, move_index_j = self.bi, (self.bj-1)
        elif (direction == Move.R):
            move_index_i, move_index_j = self.bi, (self.bj+1)
        elif (direction == Move.U):
            move_index_i, move_index_j = (self.bi-1), self.bj
        elif (direction == Move.D):
            move_index_i, move_index_j = (self.bi+1), self.bj

        move_index_value = self.Grid[move_index_i][move_index_j]

        # make a copy of
        grid = [x[:] for x in self.Grid]
        
        # swap values
        grid[move_index_i][move_index_j] = BLANK
        grid[self.bi][self.bj] = move_index_value
        # update blank index
        #grid.bi, grid.bj = move_index_i, move_index_j

        # generate new state with given move direction
        return State(self.version, grid=grid, parent=self, direction=direction)

    def f_value(self):
        return self.g_cost + self.h_cost;
    
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
        return self.f_value() < state2.f_value()
        
    def __eq__(self, state2):
        ''' Overriden equality operator to check states if they are equal based on their grid values '''
        return self.Grid == state2.Grid

############################################## Algo ##########################################

class Algo():

    def __init__(self, start, outfilePath):
        self.start = start
        self.start.g_cost = 0
        self.states_count = 0
        self.outfilePath = outfilePath
        
        
    def track_path(self, goal):
        path = []
        node = goal
        # print path
        while node.parent is not None:
            path.append(node)
            node = node.parent

        path.reverse()
        # write to file
        with open(self.outfilePath, 'w') as f:
            f.write(','.join(str(s.direction.name) for s in path))
    
        print('Total states explored: ' + str(self.states_count))
        print('Goal Depth: ' + str(goal.g_cost))

    def __gen_node(self, current, move):
        '''
        Generate new node by applying given move direction to current node
        Additionally initialize g() and h() costs
        g(): 1 + cost(current) - Actual physical cost from start to this node
        h(): Heristic function value
        '''
        new = current.move(move)
        if new:
            new.g_cost = current.g_cost + 1
            new.calculate_h()
        return new
    
    def gen_nodes(self, current):
        ret = []
        ret.append(self.__gen_node(current, Move.L))
        ret.append(self.__gen_node(current, Move.R))
        ret.append(self.__gen_node(current, Move.U))
        ret.append(self.__gen_node(current, Move.D))
        return ret
    
    ######################################### Simple A-Start ##################################
    def a_star(self):
        ''' Simple blind Breath First Search '''
        import queue as Q

        self.frontier = Q.PriorityQueue()
        self.frontier.put(self.start)
    
        self.explored = []
        while not self.frontier.empty():
            current = self.frontier.get()
            if current == GOAL:
                self.track_path(current)
                print('Goal: ' + str(current))
                return True
            self.explored.append(current)
            # generate all possible nodes from current state
            for node in self.gen_nodes(current):
                if node and node not in self.explored:
                    self.states_count += 1
                    if node == GOAL:
                        self.track_path(node)
                        print('Goal: ' + str(node))
                        return True
                    self.frontier.put(node)
        # unsolvable puzzle
        print('Puzzle not solvable: ' + str(self.start))
        return False
        
    def printFrontier(self):
        while not self.frontier.empty():
            print(self.frontier.get())
            
    ################################## Memory Bounded #######################################
    def ida_star(self):
        ''' 
        Memory bounded algorithm 
        Reference: https://algorithmsinsight.wordpress.com/graph-theory-2/ida-star-algorithm-in-general/
        '''
        self.start.calculate_h()
        threshold=self.start.h_cost

        while True:
            result = self.ida_star_driver(self.start, threshold)
            if result == True:
                return True
            elif result == MAX:
                print('Puzzle not solvable: ' + str(self.start))
                return False
            threshold = result

    def ida_star_driver(self, current, threshold):
        '''
        IDA* recursive driver function
        '''
        f = current.f_value()
        if f > threshold:
            return f
        if current == GOAL:
            self.track_path(current)
            print('IDA* Goal Found: ' + str(current))
            return True
        min = MAX
        for node in self.gen_nodes(current):
            if node:
                self.states_count += 1
                tmp = self.ida_star_driver(node, threshold)
                if tmp == True:
                    return True
                if tmp < min:
                    min = tmp
        return min
    
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
    
    start = State(puzzle, inputPath)
    print('Start State: ' + str(start))
    
    global GOAL
    GOAL = State(puzzle)
    GOAL.calculate_h()
    
    a = Algo(start, outputPath)
    if algo == 1:
        a.a_star()
    else:
        #a.a_star()
        a.ida_star()

    import resource
    def using(point=""):
        usage=resource.getrusage(resource.RUSAGE_SELF)
        return '''%s: usertime=%s systime=%s mem=%s mb
        '''%(point,usage[0],usage[1],
             (usage[2]*resource.getpagesize())/1000000.0 )

    print(using('End'))
