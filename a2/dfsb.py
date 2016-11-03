'''
CSE 537 - Assign-2
Furqan Baig (109971799)
'''
import sys
from collections import namedtuple
from collections import defaultdict

from search import DFSB
from graph import Graph
from graph import Node

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('Usage: dfsb.py <input file> <output file> <mode>')

    inputPath = sys.argv[1]
    outputPath = sys.argv[2]
    mode = int(sys.argv[3])

    if mode == 0: print('Plain DFSB')
    elif mode == 1: print('Improved DFSB')
    else: sys.exit('Invalid mode option')

    constraints = {}
    
    # get input values from file
    with open(inputPath, 'r') as f:
        lines = f.readlines()
        N = int(lines[0].split()[0])
        M = int(lines[0].split()[1])
        K = int(lines[0].split()[2])
        lines = lines[1:]
        for line in lines:
            v1 = int(line.split()[0])
            v2 = int(line.split()[1])
            if v1 in constraints:
                constraints[v1].append(v2)
            else:
                constraints[v1] = [v2]
            if v2 in constraints:
                constraints[v2].append(v1)
            else:
                constraints[v2] = [v1]

    # generate color domain
    color_domain = []
    for i in range(0, K):
        color_domain.append(i)
    # generate nodes for initial graph
    nodes = {}
    for i in range(0, N):
        # initialize all nodes with invalid color and all colors as domain
        nodes[i] = Node(i, color_domain)
    
    graph = Graph(constraints, nodes)

    print( constraints)
    #print(nodes)
    #print(graph)
    
    dfsb = DFSB(N, graph)
    dfsb.color()
