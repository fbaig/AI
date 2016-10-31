'''
CSE 537 - Assign-2
Furqan Baig (109971799)
'''
import sys
from collections import namedtuple
from collections import defaultdict

INVALID_COLOR = -1

# v1 and v2 in a given constaint can't have same colors
Constraint = namedtuple('Constraint', 'v1 v2')

class Node(object):
    
    def __init__(self, name, domain, color=INVALID_COLOR):
        self.name = name
        self.domain = list(domain)
        self.color = color

    def __repr__(self):
        return str(self.name) + '\t' + str(self.color)


    
    
class Graph:
    '''
    Index is the node number since node names (0-n)
    Value is the color of the node
    '''
    def __init__(self, constraints, nodes, node_name=None, color=None):
        self.constraints = constraints
        import copy
        self.nodes = copy.deepcopy(nodes)
        #self.nodes = dict(nodes)
        #self.nodes = nodes.copy()

        if node_name != None and color != None:
            #print('Assigning color: %d to node: %d in init'%(node, color))
            self.nodes[node_name].color = color
            
    def assignColor(self, node_name, color):
        # check for constraints
        for c in self.constraints:
            c1 = self.nodes[c.v1]
            c2 = self.nodes[c.v2]
            
            #print ('c1: ' + str(c1))
            #print ('c2: ' + str(c2))
            # 
            if c1.name == node_name and c2.color == color:
                return None
            elif c2.name == node_name and c1.color == color:
                return None

        # all is well, assign color
        return Graph(self.constraints, nodes=self.nodes, node_name=node_name, color=color)
        
        
    def isGoal(self):
        for n in self.nodes.values():
            if n.color == INVALID_COLOR:
                return False
        return True
    
    def __repr__(self):
        ret = ''
        for n in self.nodes.values():
            ret += str(n) + '\n'
        return ret
    
    def __eq__(self, other):
        #print('checking eq\n' + str(self) + str(other))
        # for n, o in zip(self.nodes.values(), other.nodes.values()):
        #     if n.color != o.color: return False
        if self.nodes == other.nodes: return True
        return False
            
        # for k in self.nodes:
        #     if self.nodes[k].color != other.nodes[k].color: return False
        # return True
            

class DFSB:

    def __init__(self, variablesCount, graph):
        self.vCount = variablesCount
        self.graph = graph

    def color(self):

        variables = list(graph.nodes.keys())
        
        # simple DFS
        frontier = []
        explored = []
        # since we start with an empty state for graph coloring, it can never be goal
        frontier.append(self.graph)
        while frontier:
            current = frontier.pop()
            explored.append(current)
            v = variables.pop()

            # print ('Curret: \n%s'%current)
            # print ('Node: %d'%v)
            # print ('Domain: %s'%self.graph.nodes[v].domain)
            
            for c in self.graph.nodes[v].domain:
                g = current.assignColor(v, c)
                #print(str(g))
                if g:
                    if g.isGoal():
                        print('Goal: \n' + str(g))
                        return True
                    elif g not in frontier and g not in explored:
                        frontier.append(g)
        
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
            constraints.append(Constraint(v1=int(line.split()[0]), v2=int(line.split()[1])))

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

    #print( constraints)
    #print(nodes)
    #print(graph)
    
    dfsb = DFSB(N, graph)
    dfsb.color()
