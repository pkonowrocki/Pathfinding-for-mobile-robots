import numpy as np
import pickle
import time
import matplotlib.pyplot as plot
import cv2 as cv
import math
from scipy.integrate import ode
def GenerateMap():
    tempmap = np.zeros([10,10])
    for i in range(np.random.randint(15,60)):
        temp = (np.random.randint(0,10), np.random.randint(0,10))
        tempmap[temp]=1
    tempmap = np.round(cv.resize(tempmap,(40,40)))
    return tempmap

def normalize(x):
    return 0.5*(np.abs(x+1) - np.abs(x-1))

def neighbours(pos, X):
    return X[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2]

def a(x):
    c = 0.01
    return -0.25*((x-c)+np.abs(x-c))

def A(pos, X):
    n = neighbours(pos, X)
    n = X[pos] - n
    n = a(n)
    n[1,1] = 1
#    n[0,0] = 0
#    n[0,2] = 0
#    n[2,0] = 0
#    n[2,2] = 0
    return n

def dx(x, u, y):
    d=np.zeros(x.shape)
    for i in range(1,x.shape[0]-1):
        for j in range(1,x.shape[1]-1):
            suma =  np.sum(np.multiply(neighbours((i,j), x), A((i,j),x)))
            d[i,j] = -x[i,j] + suma
    return d

def neighboursI( x):
    return  [(x[0]+1,x[1]),(x[0]-1,x[1]),(x[0],x[1]+1),(x[0],x[1]-1),(x[0]-1,x[1]+1),(x[0]+1,x[1]-1),(x[0]-1,x[1]-1),(x[0]+1,x[1]+1)]
    

def compute(end, start, u):
    u = GenerateMap()
    u[end] = -1
    u = np.pad(u, 1, 'constant', constant_values=0)
    x = u
    y = normalize(x)
    while x[start]>=0:
        x = x - dx(x,u,y)*0.5
        y = normalize(x+2*u)
        x = y
    res = list()
    curr = start
    while curr!=end:
        other = neighboursI(curr)
        m = curr
        for i in other:
            if(i[0]>=1 and i[0]<x.shape[0]-1 and i[1]>=1 and i[1]<x.shape[1]):
                if(x[m]>x[i]):
                    m=i
        res.append(curr)
        curr=m
    return res

if __name__ == '__main__':
    
    u = GenerateMap()
    plot.imshow(u)
    end= (0,0)
    start = (39,39)
    u[end] = -1
    u = np.pad(u, 1, 'constant', constant_values=0)
    
    fig = plot.figure(dpi=300)
    plot.imshow(u[1:41,1:41])
    fig.savefig('cnn2.png')
#    plot.imsave('cnn2.png',u[1:41,1:41])
    x = u
    y = normalize(x)
    i=0
    for _ in range(2000):
    #while x[start]>=0:
        i+=1
        x = x - dx(x,u,y)*0.05
        y = normalize(x+2*u)
        x = y
        if(i%25==24):
            plot.imshow(x)
            plot.show()
    x = y[1:41,1:41]
    xx = np.copy(u[1:41,1:41])

#    plot.imsave('cnn1.png',x)
    res = list()
    curr = start
    while curr!=end:
        other = neighboursI(curr)
        m = curr
        for i in other:
            if(i[0]>=0 and i[0]<x.shape[0] and i[1]>=0 and i[1]<x.shape[1]):
                if(x[m]>x[i]):
                    m=i
        res.append(curr)
        xx[curr]=-1
        curr=m
        print(curr)
    fig = plot.figure(dpi=300)
    plot.imshow(xx)
    fig.savefig('cnn3.png')




# Make data.
X = np.arange(0, 40)
xlen = len(X)
Y = np.arange(0,40)
ylen = len(Y)
X, Y = np.meshgrid(X, Y)
Z = x
from mpl_toolkits.mplot3d import Axes3D
fig = plot.figure(dpi=300)
ax = fig.gca(projection='3d')

from matplotlib.ticker import LinearLocator
# Create an empty array of strings with the same shape as the meshgrid, and
# populate it with two colors in a checkerboard pattern.

from matplotlib import cm
# Plot the surface with face colors taken from the array we made.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(1, -1)
ax.w_zaxis.set_major_locator(LinearLocator(10))
fig.colorbar(surf, shrink=0.5, aspect=5)
fig.savefig('cnn3d.png')
plot.show()
        
        