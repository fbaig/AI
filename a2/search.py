import sys

from graph import Graph
from graph import Node

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
            
            v = current.get_next_variable(mrv=True)

            # print ('Curret: \n%s'%current)
            # print ('Node: %d'%v)
            # print ('Domain: %s'%self.graph.nodes[v].domain)
            
            for c in self.graph.order_values_for(v): #self.graph.nodes[v].domain:
                g = current.assign_color(v, c)
                if g:
                    if g.is_goal():
                        print('Goal: \n' + str(g))
                        print ('States Explored: %d'%states_explored)
                        return True
                    # elif not g.solution_exists():
                    #     sys.exit('No solution Exists')
                    elif g not in frontier and g not in explored:
                        frontier.append(g)
        
        print ('No solution exist')

class MinConflicts:
    '''
    Local Search Minimum Conflict for Graph Map
    '''
    
