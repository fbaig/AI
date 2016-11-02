'''
CSE 537 - Assign-2
Furqan Baig (109971799)
'''
import sys
from collections import namedtuple
from collections import defaultdict

INVALID_COLOR = -1

class Node(object):
    
    def __init__(self, name, domain, color=INVALID_COLOR):
        self.name = name
        self.domain = list(domain)
        self.color = color

    def __repr__(self):
        return str(self.name) + '\t' + str(self.color) + '\t' + str(self.domain)
    
class Graph:
    '''
    Index is the node number since node names (0-n)
    Value is the color of the node
    '''
    def __init__(self, constraints, nodes, node_name=None, color=None):
        self.constraints = dict(constraints)
        import copy
        self.nodes = copy.deepcopy(nodes)

        if node_name != None and color != None:
            #print('Assigning color: %d to node: %d in init'%(node, color))
            self.nodes[node_name].color = color

        self.prune_domains()

    def solution_exists(self):
        for v in self.nodes.values():
            if not v.domain:
                return False
        return True
    
    def prune_domains(self):
        for k,v in self.constraints.items():
            if self.nodes[k].color != INVALID_COLOR and \
               self.nodes[k].color in self.nodes[v].domain:
                self.nodes[v].domain.remove(self.nodes[k].color)

            if self.nodes[v].color != INVALID_COLOR and \
               self.nodes[v].color in self.nodes[k].domain:
                self.nodes[k].domain.remove(self.nodes[v].color)
            
    def get_next_variable(self, mrv=False):
        if mrv:
            # calculate the variable with minimum remaining values
            ret = None
            min_domain_size = sys.maxsize
            for k,v in self.nodes.items():
                if v.color == INVALID_COLOR and len(v.domain) < min_domain_size:
                    ret = k
                    min_domain_size = len(v.domain)
            return ret
        else:
            # simply return the next variable which does not have a valid color
            for k,v in self.nodes.items():
                if v.color == INVALID_COLOR: return k
        return None
    
    def assignColor(self, node_name, color):
        # check for constraints
        for k,v in self.constraints.items():
            if k == node_name and self.nodes[v].color == color:
                return None
            elif v == node_name and self.nodes[k].color == color:
                return None

        # all is well, assign color
        g = Graph(self.constraints, nodes=self.nodes, node_name=node_name, color=color)
        if g.solution_exists(): return g
        return None
        
        
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
        if self.nodes == other.nodes: return True
        return False

class DFSB:

    def __init__(self, variablesCount, graph):
        self.vCount = variablesCount
        self.graph = graph
        
    def color(self):

        states_explored = 0
        # simple DFS
        frontier = []
        explored = []
        # since we start with an empty state for graph coloring, it can never be goal
        frontier.append(self.graph)
        while frontier:
            states_explored +=1
            current = frontier.pop()
            explored.append(current)
            
            v = current.get_next_variable(mrv=False)

            # print ('Curret: \n%s'%current)
            # print ('Node: %d'%v)
            # print ('Domain: %s'%self.graph.nodes[v].domain)
            
            for c in self.graph.nodes[v].domain:
                g = current.assignColor(v, c)
                if g:
                    if g.isGoal():
                        print('Goal: \n' + str(g))
                        print ('States Explored: %d'%states_explored)
                        return True
                    # elif not g.solution_exists():
                    #     sys.exit('No solution Exists')
                    elif g not in frontier and g not in explored:
                        frontier.append(g)
        
        print ('No solution exist')

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
            constraints[int(line.split()[0])] = int(line.split()[1])
            #constraints.append(Constraint(v1=int(line.split()[0]), v2=int(line.split()[1])))

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
