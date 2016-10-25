'''
CSE 537 - Assign-2
Furqan Baig (109971799)
'''
import sys
from collections import namedtuple

INVALID_COLOR = -1

# v1 and v2 in a given constaint can't have same colors
Constraint = namedtuple('Constraint', 'v1 v2')

class Node:
    def __init__(self, name, color):
        self.name = name
        self.color = color

    def __repr__(self):
        return str(self.name) + '\t' + str(self.color)
        
class Graph:
    def __init__(self, variables):
        self.nodes = []
        for i in range(0, variables):
            # initialize all nodes with invalid color
            self.nodes.append(Node(i, INVALID_COLOR))

    def __repr__(self):
        ret = ''
        for n in self.nodes:
            ret += str(n) + '\n'
        return ret

class DFSB:

    def __init__(self, variablesCount, constraints, colors):
        self.vCount = variablesCount
        self.constraints = constraints
        self.colors = colors
        self.graph = Graph(variablesCount)

    def color(self):
        print (str(self.graph))
        print ('bla')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('Usage: dfsb.py <input file> <output file> <mode>')

    inputPath = sys.argv[1]
    outputPath = sys.argv[2]
    mode = int(sys.argv[3])

    if mode == 0: print('Plain DFSB')
    elif mode == 1: print('Improved DFSB')
    else: sys.exit('Invalid mode option')

    constraints = []
    
    # get input values from file
    with open(inputPath, 'r') as f:
        lines = f.readlines()
        N = int(lines[0].split()[0])
        M = int(lines[0].split()[1])
        K = int(lines[0].split()[2])
        lines = lines[1:]
        for line in lines:
            constraints.append(Constraint(v1=line.split()[0], v2=line.split()[1]))


    dfsb = DFSB(N, constraints, K)
    dfsb.color()
    for c in constraints:
        print (c)
