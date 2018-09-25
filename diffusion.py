import numpy as np
import cv2 as cv
import sys
class Diffusion:
    def imread(self,path):
        self.image =  cv.threshold(cv.bitwise_not(cv.imread(path,0)), 127, 255, cv.THRESH_BINARY)[1]
        self.tab={}
        return self.image
    
    def startend(self, start, end):
        self.start = (start[1],start[0])
        self.end = (end[1],end[0])
        
    def init(self, matrix):
        self.matrix = matrix
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                self.tab[(i,j)] = sys.maxsize
        self._maxvalue = matrix.shape[0]*matrix.shape[1]+1
        self.donetab = self.tab.copy()
        self.tab[self.end_cell] = 0
        self.donetab[self.end_cell] = 0
        
    def discretize(self, n, m):
        self.discrete = np.ndarray([n,m])
        self.n = n
        self.m = m
        self.cell_n = int(self.image.shape[0]/n)
        self.cell_m = int(self.image.shape[1]/m)
        self.start_cell = (np.floor(self.start[0]/self.cell_n),np.floor(self.start[1]/self.cell_n))
        self.end_cell = (np.floor(self.end[0]/self.cell_m),np.floor(self.end[1]/self.cell_m))
        if((n,m)==self.image.shape):
            self.discrete = self.image
            return self.discrete
        kernel = np.ones([self.cell_n,self.cell_m])
        for i in range(n):
            for j in range(m):
                self.discrete[i][j] = np.sum(np.multiply(kernel, self.image[self.cell_n*i:self.cell_n*(i+1),self.cell_m*j:self.cell_m*(j+1)]))
        self.discrete = cv.threshold(self.discrete, 1, 255, cv.THRESH_BINARY)[1]
        return self.discrete

    def score(self, x):
        for y in self.neighbours(x):
            if(y[0]>=0 and y[0]<self.matrix.shape[0] and y[1]>=0 and y[1]<self.matrix.shape[1] and self.free(y)):  
                if(self.donetab[y]>self.donetab[x]+5):
                    self.donetab[y]=self.donetab[x]+5
                    self.tab[y]=self.donetab[y]
        for y in self.neighbours_cross(x):
            if(y[0]>=0 and y[0]<self.matrix.shape[0] and y[1]>=0 and y[1]<self.matrix.shape[1] and self.free(y)):  
                if(self.donetab[y]>self.donetab[x]+7):
                    self.donetab[y]=self.donetab[x]+7
                    self.tab[y]=self.donetab[y]
                
    def path(self):
        res =list()
        res.append(self.start)
        curr = self.start_cell
        while curr!=self.end_cell:
            least=curr
            neigh=list()
            for y in self.neighbours(curr):
                if(y[0]>=0 and y[0]<self.matrix.shape[0] and y[1]>=0 and y[1]<self.matrix.shape[1] ):
                   neigh.append(y) 
            for y in self.neighbours_cross(curr):
                if(y[0]>=0 and y[0]<self.matrix.shape[0] and y[1]>=0 and y[1]<self.matrix.shape[1] ):
                    neigh.append(y)
            least = min(neigh, key= lambda q: self.donetab[q])
            res.append((int(self.cell_n*(least[0]+0.5)),int(self.cell_m*(least[1]+0.5))))
            #res.append(least)
            curr = least
        res.append(self.end)
        return res
                
        
    def free(self, x):
        return self.matrix[x]==0

    def neighbours(self, x):
        return  [(x[0]+1,x[1]),(x[0]-1,x[1]),(x[0],x[1]+1),(x[0],x[1]-1)]
    
    def neighbours_cross(self,x):
        return [(x[0]-1,x[1]+1),(x[0]+1,x[1]-1),(x[0]-1,x[1]-1),(x[0]+1,x[1]+1)]
    
    def diffuse(self):
        while len(self.tab)>0:
            curr = min(self.tab.items(), key= lambda q: q[1])[0]    
            self.score(curr)
            self.tab.pop(curr)
    
    def tab2matrix(self):        
        n = np.ndarray([self.matrix.shape[0],self.matrix.shape[1]])
        for p in self.donetab.keys():
            n[p[0],p[1]]=self.donetab[p]
#            if(self._maxvalue!=self.donetab[p]):
#                n[p[0],p[1]]=self.donetab[p]
#            else:
#                n[p[0],p[1]]=0
        return n
    
    def dicretized_im(self):
        n = np.ndarray(self.image.shape)
        for i in range(self.n):
            for j in range(self.m):
                n[self.cell_n*i:self.cell_n*(i+1),self.cell_m*j:self.cell_m*(j+1)] = self.discrete[i,j]
        return n