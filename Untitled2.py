#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
#from scipy.sparse.linalg import eigs

def laplacian2D(N):
    diag=np.ones([N*N])
    mat=sp.spdiags([diag,-2*diag,diag],[-1,0,1],N,N)
    I=sp.eye(N)
    return sp.kron(I,mat,format='csr')+sp.kron(mat,I)


def ddx2D(N):
    diag=np.ones([N*N])
    mat=sp.spdiags([-diag,0*diag,diag],[-1,0,1],N,N)
    I=sp.eye(N)
    return sp.kron(I,mat,format='csr')

def ddy2D(N):
    diag=np.ones([N*N])
    mat=sp.spdiags([-diag,0*diag,diag],[-1,0,1],N,N)
    I=sp.eye(N)
    return sp.kron(mat,I)



N = 100
dx = 1/N
x = np.linspace(dx,1-dx,N-1)
y = np.linspace(dx,1-dx,N-1)
x, y = np.meshgrid(x,y)
pi = np.pi

  
lap = laplacian2D(N-1)/dx**2
ddx = ddx2D(N-1)/(2*dx)
ddy = ddy2D(N-1)/(2*dx)
#lap = lap.toarray()

r = 0.1
beta = 1
nu = 0.001

A = r * lap + beta * ddx
B = - nu * lap.dot(lap) - beta * ddx
f = 0.005*np.sin(pi*y)
f_long = np.reshape(f,(N-1)**2)

niter=8
fig, ax = plt.subplots(niter, figsize = (20, 20))

psi_long = np.zeros((N-1)**2)
for i in range(niter):
    q_long = lap.dot(psi_long)
    q_long_x = ddx.dot(q_long)
    q_long_y = ddy.dot(q_long)
    psi_long_x = ddx.dot(psi_long)
    psi_long_y = ddy.dot(psi_long)
    rhs_long = f_long - (psi_long_x*q_long_y - psi_long_y*q_long_x)
    
    
    #psi_long = spsolve(A,rhs_long)
    psi_long = spsolve(B,rhs_long)
    
    psi = np.reshape(psi_long,(N-1,N-1)) 
    ax[i].contourf(x,y,psi)
 




# psi = np.sin(5*pi*x)*np.sin(3*pi*y)


# psi_long = np.reshape(psi,((N-1)**2,1))

# #q_long = lap.dot(psi_long)
# q_long = ddy.dot(psi_long)

# q = np.reshape(q_long,(N-1,N-1))

# fig, ax = plt.subplots(2)

# ax[0].contourf(x,y,psi)
# ax[1].contourf(x,y,q)

#print(np.max(q)/pi)

#print(lap.toarray())
#vals, vecs = eigs(lap,k=(N-1)**2)
#print(vals)
#print(vecs)


# In[8]:


epsilon = 0.01 
A = 1
C_1 = 4
C_2 = 5

def psi_approx(x, y):
    return (1 - x - np.exp(-x/epsilon)) * A * np.sin(np.pi * y)

def psi_exact(x, y):
    return np.sin(np.pi * y) * (C_1 * np.exp((-1 + np.sqrt(1 + 4 * epsilon**2 * np.pi**2))*x) + C_2 * np.exp((-1 - np.sqrt(1 + 4 * epsilon**2 * np.pi**2))*x) - A/(np.pi**2*epsilon))
    
N = 100
x = np.linspace(0,1,N)
y = np.linspace(0,1,N)
x, y = np.meshgrid(x,y)

Psi_approx = psi_approx(x, y)
Psi_exact = psi_exact(x, y)

print(Psi_exact.shape)

fig, ax = plt.subplots(2, figsize = (10, 20))
ax[0].contourf(x, y, Psi_approx)
ax[1].contourf(x, y, Psi_exact)

plt.show()


# In[ ]:




