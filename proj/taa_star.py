'''
Furqan Baig (109971799)
Usama Mahmood
Ubaid Ullah Hafeez

Incremental Heuristic Search (Tree Adaptive A-Star)
'''
DEfAULT_COST = 1
INFINITY = 99999999

class State:
    def __init__(self, x, y):
        '''
        generated: The last A* star search that generated this state
        '''
        self.loc = (x, y)
        
        self.generated = 0
        self.id = 0
        self.reusable_tree = None
        self.search_tree = None
        
        self.reinit(counter=1)

    def reinit(self, counter):
        '''
        counter: Current A* search number
        '''
        if self.generated == 0:
            state.g = INFINITY
            state.h = 0 # TODO: user provided heuristic function
        elif self.generated != counter:
            state.g = INFINITY
        state.generated = self.counter
        
    def __lt__(self, other):
        return ((self.g+self.h) < (other.g+other.h))
    
    def __eq__(self, other):
        if self.loc == other.loc: return True
        return False

class Path:
    def __init__(self, states):
        self.states = []
        self.Hmax = -1
        self.Hmin = -1
        
    
class TAA_Star:
    def __init__(self, start_x, start_y, goal_x, goal_y):
        
        self.start = State(start_x, start_y)
        self.goal = State(goal_x, goal_y)
        
        self.paths = []

    def find_path(self):
        self.counter = 1
        # Hmax[0] = -1
        while self.start != self.goal:
            self.start.g = 0
            import Queue
            self.open_list = Queue.PriorityQueue()
            self.close_list = []

            self.open_list.append(self.start)
            if self.compute_path() == False:
                print ('Goal state is unreachable')
                return False

            self.counter += 1
        return True
            
        
    def add_path(self, state):
        if state != self.goal:
            self.paths.append()


    def remove_path(self, state):
        x = state.id


    def get_succ(self, state):
        # TODO compute successors of given state
        return []
    
    def compute_path(self):

        while self.open_list:
            # remove state from open list with smallest (g+h) value
            s = self.open_list.pop()
            if s == self.goal or s.h <= Hmax[s.id]:
                # s is in reusable tree
                for s_dash in self.closed:
                    s_dash.h = s.g + s.h - s_dash.g
                self.add_path(s)
                return True
            
            self.closed.append(s)
            for succ in self.get_succ(s):
                succ.reinit(counter=self.counter)
                if succ.g > s.g + cost(s, succ):
                    succ.g = s.g + cost(s, succ)
                    succ.search_tree = s
                    if succ in self.open_list:
                        self.open_list.remove(succ)
                    self.open_list.append(succ)
                    
        return False
