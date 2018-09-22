import imageio as img
import matplotlib.pyplot as plot
import numpy as np

class Diffusion:
    def imread(self,path):
        self.image = img.imread(path)[:,:,0]
        self.tab={}
        
    def init(self, matrix, start, end):
        self.matrix = matrix
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                self.tab[(i,j)] = matrix.shape[0]*matrix.shape[1]+1
        self._maxvalue = matrix.shape[0]*matrix.shape[1]+1
        self.donetab = self.tab.copy()
        self.start = start
        self.end = end
        self.tab[end] = 0
        self.donetab[end] = 0
        

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
                

    def free(self, x):
        return self.matrix[x[0],x[1]]>200

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
            if(self._maxvalue!=self.donetab[p]):
                n[p[0],p[1]]=self.donetab[p]
            else:
                n[p[0],p[1]]=0
        return n