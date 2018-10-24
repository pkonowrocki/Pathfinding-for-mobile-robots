import numpy as np
import cv2 as cv
import sys
class ThetaStar:
    def theta(self):
        self.gScore={}
        self.gScore[self.start_cell] = 0
        self.parent = {}
        self.parent[self.start_cell] = self.start_cell
        self.o = {}
        self.o[self.start_cell] = self.gScore[self.start_cell]+self.heuristic(self.start_cell)
        self.c={}
        while len(self.o)>0:
            s=min(self.o.keys(),key = lambda x:self.o[x])#self.o.popitem()[0]
            self.o.pop(s)
            if s==self.end_cell:
                self.res = list()
                return self.reconstruct_path(s)
            self.c[s]=None
            for y in self.neighbours(s):
                if(y[0]>=0 and y[0]<self.discrete.shape[0] and y[1]>=0 and y[1]<self.discrete.shape[1] and self.free(y)):
                    if y not in self.c.keys():
                        if y not in self.o.keys():
                            self.gScore[y] = sys.maxsize
                            self.parent[y] = None
                        self.update_vertex(s,y)
            for y in self.neighbours_cross(s):
                if(y[0]>=0 and y[0]<self.discrete.shape[0] and y[1]>=0 and y[1]<self.discrete.shape[1] and self.free(y)):
                    if y not in self.c.keys():
                        if y not in self.o.keys():
                            self.gScore[y] = sys.maxsize
                            self.parent[y] = None
                        self.update_vertex(s,y)
        return None
    
    def line_of_sight(self, x, y):        
        x = ((x[0]+0.5)*self.cell_n,(x[1]+0.5)*self.cell_m)
        y = ((y[0]+0.5)*self.cell_n,(y[1]+0.5)*self.cell_m)
        if(x[0]==y[0]):
            a=min([x[1],y[1]])
            b=max([x[1],y[1]])+1
            for i in range(int(a),int(b)):
                if(self.image[int(x[0]),int(i)]):
                    return False
        else:
            a=min([x[0],y[0]])
            b=max([x[0],y[0]])+1
            if(a==x[0]):
                m=x[1]
            else:
                m=y[1]
            for i in range(int(a),int(b)):
                c=np.floor(  (y[1]-x[1])/(y[0]-x[0])*(i-a)+m)
                if(c<self.discrete.shape[1] and self.image[int(i),int(c)]):
                    return False
        return True
       
        
    def update_vertex(self, s, y):
        if(self.line_of_sight(self.parent[s], y)):
            if(self.gScore[self.parent[s]] + self.dist(self.parent[s],y) <self.gScore[y]):
                self.gScore[y]=self.gScore[self.parent[s]] + self.dist(self.parent[s],y)
                self.parent[y] = self.parent[s]
                if y in self.o:
                    self.o.pop(y)
                self.o[y] = self.gScore[y]+self.heuristic(y)
        else:
            if(self.gScore[s] + self.dist(s,y) <self.gScore[y]):
                self.gScore[y]=self.gScore[s] + self.dist(s,y)
                self.parent[y] = s
                if y in self.o:
                    self.o.pop(y)
                self.o[y] = self.gScore[y]+self.heuristic(y)
    
    def reconstruct_path(self,s):
        curr = s
        result = list()
        #result.append(self.end[::-1])
        while(curr!=self.parent[curr]):
            result.append((int(self.cell_m*(curr[1]+0.5)),int(self.cell_n*(curr[0]+0.5))))
            curr = self.parent[curr]
        result.append((int(self.cell_m*(curr[1]+0.5)),int(self.cell_n*(curr[0]+0.5))))
        #result.append(self.start[::-1])
        return result
        
        
    def neighbours(self, x):
        return  [(x[0]+1,x[1]),(x[0]-1,x[1]),(x[0],x[1]+1),(x[0],x[1]-1)]
    
    def neighbours_cross(self,x):
        return [(x[0]-1,x[1]+1),(x[0]+1,x[1]-1),(x[0]-1,x[1]-1),(x[0]+1,x[1]+1)]
    
    def dist(self, x,y):
        return abs(self.cell_n*(x[0]-y[0]))+abs(self.cell_n*(x[1]-y[1])) 
    
    def heuristic(self, x):
        return abs(x[0]-self.end_cell[0])+abs(x[1]-self.end_cell[1])
    
    
    def imread(self,path):
        self.image =  cv.threshold(cv.bitwise_not(cv.imread(path,0)), 127, 255, cv.THRESH_BINARY)[1]
        self.tab={}
        return self.image
    
    def startend(self, start, end):
        self.start = start[::-1]
        self.end = end[::-1]
    
    def free(self, x):
        return self.discrete[int(x[0]),int(x[1])]==0
        
    def discretize(self, n, m):
        self.discrete = np.ndarray([n,m])
        self.n = n
        self.m = m
        self.cell_n = int(self.image.shape[0]/n)
        self.cell_m = int(self.image.shape[1]/m)
        self.start_cell = (np.floor(self.start[0]/self.cell_n),np.floor(self.start[1]/self.cell_n))
        self.end_cell = (np.floor(self.end[0]/self.cell_m),np.floor(self.end[1]/self.cell_m))
        kernel = np.ones([self.cell_n,self.cell_m])
        for i in range(n):
            for j in range(m):
                self.discrete[i][j] = np.sum(np.multiply(kernel, self.image[self.cell_n*i:self.cell_n*(i+1),self.cell_m*j:self.cell_m*(j+1)]))
        self.discrete = cv.threshold(self.discrete, 1, 255, cv.THRESH_BINARY)[1]
        return self.discrete
    
    
    def tab2matrix(self):        
        n = np.ndarray([self.matrix.shape[0],self.matrix.shape[1]])
        for p in self.donetab.keys():
            n[p[0],p[1]]=self.donetab[p]
        return n
    
    def dicretized_im(self):
        n = np.ndarray(self.image.shape)
        for i in range(self.n):
            for j in range(self.m):
                n[self.cell_n*i:self.cell_n*(i+1),self.cell_m*j:self.cell_m*(j+1)] = self.discrete[i,j]
        return n
    