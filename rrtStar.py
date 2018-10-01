import numpy as np
import cv2 as cv
import sys

class RRTstar:
    def rrtStar(self, N):
        self.Nodes = set()
        #self.Nodes.add(self.end)
        self.T = {}
        self.cost = {}
        self.cost[self.start] = 0
        self.insert_node(self.start, None)
        while self.end not in self.Nodes:
            z_rand = self.sampling()
            z_nearest = self.nearest(z_rand)
            z_new = self.steer(z_nearest,z_rand)
            self.cost[z_new] = self.cost[z_nearest]+self.dist(z_nearest,z_new)
            
            if(self.line_of_sight(z_new,z_nearest)):
                self.Nodes.add(z_new)
                z_min = z_nearest
                Z_near = self.near(z_new)
                
                for z_near in Z_near:
                    if(self.line_of_sight(z_near,z_new)):
                        c = self.cost[z_near] + self.dist(z_near, z_new)
                        if(c<self.cost[z_new]):
                            z_min=z_near
                    
                            
                self.insert_node(z_min,z_new)
             
                self.rewire(Z_near,z_new, z_min)
            if(self.dist(z_new,self.end)<self.dq and self.line_of_sight(z_new, self.end)):
               self.insert_node(z_new,self.end)
               return self.T
                        
                
        return self.T
        
    def rewire(self, Z_near, z_new, z_min):
        for x_near in Z_near:
           if(x_near!=z_min and self.line_of_sight(x_near, z_new) and self.cost[x_near]> self.cost[z_new] + self.dist(z_new,x_near)):
               x_parent = self.parent(x_near)
               self.delete_node(x_parent, x_near)
               self.insert_node(z_new, x_near)
               
    def delete_node(self, parent, child):
        self.T[parent].remove(child)
           
    def parent(self, x_child):
        for key in self.T:
            if x_child in self.T[key]:
                return key
    
    def path(self):
        res = list()
        curr = self.end
        
        while(curr!=self.start):
            res.append(curr)
            curr = self.parent(curr)
        res.append(self.start)
        return res
        
    def near(self, z):
        result = set()
        for i in self.Nodes:
            if(self.dist(z,i)<20):
                result.add(i)
        return result
        
    def steer(self, z_nearest, z_rand):
        self.dq = 50
        if(z_rand==z_nearest):
            return z_rand
        z_new = (int(self.dq*(z_rand[0]-z_nearest[0])/self.dist(z_rand,z_nearest))+z_nearest[0],
                 int(self.dq*(z_rand[1]-z_nearest[1])/self.dist(z_rand,z_nearest))+z_nearest[1])
        if(self.free(z_new)):#and z_new[0]>=0 and z_new[0]<self.image.shape[0] and z_new[1]>=0 and z_new[1]<self.image.shape[1]):
            return z_new
        else:
            return self.steer(z_nearest, self.sampling()) 
        
    def insert_node(self, parent, child):
        if(parent not in self.T.keys()):
            self.T[parent]=set()
        if(child!=None):
            self.T[parent].add(child)
            self.cost[child] = self.cost[parent]+self.dist(parent,child)
            self.Nodes.add(child)
        self.Nodes.add(parent)
    
    def nearest(self, z):
        n = self.start
        ndist = self.dist(z,n)
        for i in self.Nodes:
            if(self.dist(z,i)<ndist):
                n = i
                ndist = self.dist(z,i)   
        return n
        
    def line_of_sight(self, x, y):
        if(x[0]>y[0]):
            temp = y[0]
            y=(x[0],y[1])
            x = (temp,x[1])
        if(x[1]>y[1]):
            temp = y[1]
            y = (y[0],x[1])
            x = (x[0],temp)
        kernel = np.zeros([int(abs(y[0]-x[0]))+1,int(abs(y[1]-x[1]))+1])
        cv.line(kernel,(0,0),tuple(kernel.shape)[::-1],1,1)
        if(kernel.shape[0]>0 and kernel.shape[1]>0):
            kernel[0,0]=1
            kernel[kernel.shape[0]-1,kernel.shape[1]-1]=1
            
            if(np.sum(np.multiply(kernel, self.image[int(x[0]):int(y[0]+1),int(x[1]):int(y[1]+1)]))>0):
                return False
            else:
                return True
        return True
    
    def dist(self, x,y):
        return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
    
    def imread(self,path):
        self.image =  cv.threshold(cv.bitwise_not(cv.imread(path,0)), 127, 255, cv.THRESH_BINARY)[1]
        return self.image
    
    def startend(self, start, end):
        self.start = (start[1],start[0])
        self.end = (end[1],end[0])
    
    def free(self, x):
        if(x[0]>=0 and x[0]<self.image.shape[0] and x[1]>=0 and x[1]<self.image.shape[1]):
            return self.image[int(x[0]),int(x[1])]==0
        else:
            return False
        
    def sampling(self):
#        x = (int(np.random.randint(0,self.image.shape[0])),
#             int(np.random.randint(0,self.image.shape[1])))
        x = (int(np.random.normal(self.end[0],self.image.shape[0]/2)),
             int(np.random.normal(self.end[1],self.image.shape[1]/2)))
#        while(not (x[0]>=0 and x[0]<self.image.shape[0] and x[1]>=0 and x[1]<self.image.shape[1])):
#            x = (int(np.random.randint(0,self.image.shape[0])),
#             int(np.random.randint(0,self.image.shape[1])))
        return x
    
    
   