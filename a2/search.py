import sys

from graph import Graph
from graph import Node

class DFSB:

    def __init__(self, graph, simple):
        self.graph = graph
        self.simple = simple
        
    def color(self):

        states_explored = 0
        # simple DFS
        import collections
        frontier = collections.deque()
        explored = collections.deque() 
        # since we start with an empty state for graph coloring, it can never be goal
        frontier.append(self.graph)
        while frontier:
            states_explored +=1
            current = frontier.pop()
            explored.append(current)
            if self.simple:
                v = current.get_next_variable()
                color_values = self.graph.order_values_for(v)
            else:
                v = current.get_next_variable(mrv=True)
                color_values = self.graph.order_values_for(v, lcv=False)
            
            for c in color_values:
                g = current.assign_color(v, c)
                if g:
                    if g.is_goal():
                        print('Goal: \n' + str(g))
                        print ('States Explored: %d'%states_explored)
                        return True
                    elif g not in frontier and g not in explored:
                        frontier.append(g)

        print ('No solution exist')

class MinConflicts:
    '''
    Local Search Minimum Conflict for Graph Map
    '''
    def __init__(self, graph):
        self.graph = graph
        self.assign_random_colors()

    def color(self):
        max_iterations = 100
        for i in range(0, max_iterations):
            if self.graph.is_goal(check_conflicts=True):
                print ('GOAL:')
                print (self.graph)
            else:
                print ('Not goal')
        
    def assign_random_colors(self):
        # assign random values to all nodes
        for k,v in self.graph.nodes.items():
            # select a random color from domain of current node
            from random import randint
            color = randint(0, len(v.domain)-1)
            v.color = color

        print (self.graph)
        print (self.graph.count_conflicts())
