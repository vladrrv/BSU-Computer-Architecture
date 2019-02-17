
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from jupyterthemes import jtplot
jtplot.style()


# In[6]:


def u(x, y):
    return np.sin(x+1)*x - np.cos(y+1)*y


Z = np.loadtxt('u.txt')
n = Z.shape[0]
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)
Z0 = u(Y, X)




# In[10]:


fig = plt.figure(figsize=(8,4), dpi=100)
ax = fig.add_axes((0, 1, 0.5, 1), projection='3d')
ax.plot_surface(X, Y, Z0, rstride=1, cstride=1, cmap='viridis')
ax.set_title('Ground True');
ax = fig.add_axes((0.5, 1, 0.5, 1), projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
ax.set_title('Approximate');
fig.savefig('v.png')

