class point:
    def __init__(self,x,y,weight=9999999):
        self.x=x
        self.y=y
        self.weight=weight
    def __eq__(self,other):
        return (self.x==other.x and self.y==other.y)
    def __hash__(self):
        return hash((self.x,self.y))
class graph:
    def __init__(self):
        self.graph={}
    def add_node(self,A):
        self.graph[A]=set()
    def add_vert(self,A,B):
        if (A not in self.graph.keys()):
            self.add_node(A)
        if (B not in self.graph.keys()):
            self.add_node(B)
        self.graph[A].add(B)
        self.graph[B].add(A)