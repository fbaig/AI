
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
    def __init__(self, constraints, nodes, node_name=None, color=None, simple_dfs=True):
        self.simple_dfs = simple_dfs
        # constraints are readonly, no need for deepcopy
        self.constraints = constraints
        import copy
        self.nodes = copy.deepcopy(nodes)
        
        if node_name != None and color != None:
            self.nodes[node_name].color = color
            # remove all other colors from domain of this node
            self.nodes[node_name].domain = [color]

        if not self.simple_dfs:
            self._prune_domains()

    def _ac3(self):
        '''
        Algorithm followed from Wikipedia
        '''
        import collections
        worklist = collections.deque()
        for k,v in self.constraints.items():
            for n in v:
                if (k,n) not in worklist and (n,k) not in worklist:
                    worklist.append((k,n))
        
        while worklist:
            arc = worklist.pop()
            if self._ac3_arc_reduce(arc[0], arc[1]):
                if not self.nodes[arc[0]].domain:
                    return False
        return True
    
    def _ac3_arc_reduce(self, x, y):
        change = False
        for vx in self.nodes[x].domain:
            found = False
            # find a consistent value in domain of y
            for vy in self.nodes[y].domain:
                # check if vx and vy are consistent i.e. not equal
                if vx != vy:
                    found = True
                    break
            # if no consistent values found
            if not found:
                #print('&&&&&&&&&&&&&&&&&&&& CHANGING &&&&&&&&&&&&&&&&&&&&&')
                self.nodes[x].domain.remove(vx)
                change = True
        return change
    
    def _prune_domains(self):
        '''
        Remove colors from domains of neighboring variables having valid 
        color assignment
        '''
        for k,v in self.constraints.items():
            node_color = self.nodes[k].color
            if node_color != INVALID_COLOR:
                # remove color from all neighboring nodes
                for n in v:
                    if node_color in self.nodes[n].domain:
                        self.nodes[n].domain.remove(node_color)
            
    def get_next_variable(self, mrv=False):
        '''
        Order variables either naivley on availability or using 
        Minimum Remaining Value (MRV)
        '''
        if mrv:
            # calculate the variable with minimum remaining values
            ret = None
            import sys
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

    def order_values_for(self, variable, lcv=False):
        '''
        Order values for given variable either naively or 
        based on Least Constraining Value (LCV)
        '''
        if lcv:
            raise NotImplementedError()
        else:
            return self.nodes[variable].domain
    
    def assign_color(self, node_name, color):
        '''
        Returns a new instance of graph after assigning given color to given 
        node if possible (based on constraints)
        '''
        # check for constraints
        for k,v in self.constraints.items():
            if k == node_name:
                # check for each neighbor
                for n in v:
                    if self.nodes[n].color == color:
                        return None

        # all is well, assign color
        g = Graph(self.constraints, \
                  nodes=self.nodes, \
                  node_name=node_name, \
                  color=color, \
                  simple_dfs = self.simple_dfs)
        
        if self.simple_dfs: return g
        elif g._ac3(): return g
        return None
        

    def count_conflicts(self):
        '''
        Count number of inconsistent color assignments according to constraints
        '''
        conflicts = 0
        from collections import deque
        counted = deque()
        for k,neighbors in self.constraints.items():
            for n in neighbors:
                if (k,n) not in counted and (n,k) not in counted:
                    counted.append((k,n))
                    if self.nodes[k].color == self.nodes[n].color:
                        conflicts += 1
        return conflicts
    
    def is_goal(self, check_conflicts=False):
        '''
        If all nodes in the graph have valid colors assigned
        '''
        
        for n in self.nodes.values():
            if n.color == INVALID_COLOR:
                return False
        if check_conflicts:
            if self.count_conflicts() != 0:
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
