import numpy as np
import cv2 as cv
import sys
class Diffusion:
    def diffuse(self):
        self.c = set()
        self.o = set()
        self.cameFrom = {}
        gScore={}
        gScore[self.start_cell] = 0
        self.o.add(self.start_cell)
        while (len(self.o)>0):
            curr = min(self.o, key = lambda x: gScore[x])
            if(curr==self.end_cell):
                return self.reconstruct_path()
            self.o.remove(curr)
            self.c.add(curr)
            for y in self.neighbours(curr):
                if(y[0]>=0 and y[0]<self.discrete.shape[0] and y[1]>=0 and y[1]<self.discrete.shape[1] and self.free(y)):
                    if y in self.c:
                        continue
                    tentative_gScore = gScore[curr] + self.dist(curr,y)
                    if y not in self.o:
                        self.o.add(y)
                        gScore[y] = sys.maxsize
                    elif (tentative_gScore >= gScore[y]):
                        continue
                    self.cameFrom[y] = curr
                    gScore[y] = tentative_gScore

        
        
    
    
    def reconstruct_path(self):
        curr = self.end_cell
        result = list()
        result.append(self.end[::-1])
        while(curr!=self.start_cell):
            result.append((int(self.cell_m*(curr[1]+0.5)),int(self.cell_n*(curr[0]+0.5))))
            curr = self.cameFrom[curr]
        result.append((int(self.cell_m*(curr[1]+0.5)),int(self.cell_n*(curr[0]+0.5))))
        result.append(self.start[::-1])
        return result
        
        
    def neighbours(self, x):
        return  [(x[0]+1,x[1]),(x[0]-1,x[1]),(x[0],x[1]+1),(x[0],x[1]-1),(x[0]-1,x[1]+1),(x[0]+1,x[1]-1),(x[0]-1,x[1]-1),(x[0]+1,x[1]+1)]
    
    def dist(self, x,y):
        return abs(self.cell_n*(x[0]-y[0]))+abs(self.cell_n*(x[1]-y[1])) 

    
    def imread(self,path):
        self.image =  cv.threshold(cv.bitwise_not(cv.imread(path,0)), 127, 255, cv.THRESH_BINARY)[1]
        self.tab={}
        return self.image
    
    def startend(self, start, end):
        self.start = (start[1],start[0])
        self.end = (end[1],end[0])
    
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
        n = np.ndarray([self.discrete.shape[0],self.discrete.shape[1]])
        for p in self.donetab.keys():
            n[p[0],p[1]]=self.donetab[p]
        return n
    
    def dicretized_im(self):
        n = np.ndarray(self.image.shape)
        for i in range(self.n):
            for j in range(self.m):
                n[self.cell_n*i:self.cell_n*(i+1),self.cell_m*j:self.cell_m*(j+1)] = self.discrete[i,j]
        return n
    