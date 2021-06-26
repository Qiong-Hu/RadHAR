#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


import csv
import numpy as np
import pandas as pd
import math
import time


# In[3]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# # Read data

# In[5]:


filename = "akash_home_walk_1.txt"


# In[29]:


def Get_data(filename):
    with open(filename) as f:
        lines = f.readlines()
    
    frame_num_count = -1
    frame_num = []
    x = []
    y = []
    z = []
    velocity = []
    intensity = []
    depth = []
    
    # Offset
    x_off = 0
    y_off = 4
    z_off = 8
    velocity_off = 24
    intensity_off = 16
    depth_off = 20
    point_step = 32
    
    for line in lines:
        # Find where data frames are
        if line[:5] == 'data:':
            frame_num_count += 1
            frame_data = line[7:-2].split(",")
            frame_data = np.asarray(frame_data)
            frame_data = frame_data.astype(np.uint8)
            
            init_pt1 = 0
            init_pt2 = point_step
            # Look at each point in the frame
            while init_pt2 <= len(frame_data):
                pt = frame_data[init_pt1:init_pt2]
                # Convert uint8 to float32
                x.append(float(pt[x_off:x_off+4].view('<f4')))
                y.append(float(pt[y_off:y_off+4].view('<f4')))
                z.append(float(pt[z_off:z_off+4].view('<f4')))
                velocity.append(float(pt[velocity_off:velocity_off+4].view('<f4')))
                intensity.append(float(pt[intensity_off:intensity_off+4].view('<f4')))
                depth.append(float(pt[depth_off:depth_off+4].view('<f4')))
                frame_num.append(frame_num_count)
                
                init_pt1 += point_step
                init_pt2 += point_step
    
    frame_num = np.asarray(frame_num)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    velocity = np.asarray(velocity)
    intensity = np.asarray(intensity)

    data = pd.DataFrame()
    data['frame_num'] = frame_num.astype(np.int)
    data['x'] = x.astype(np.float)
    data['y'] = y.astype(np.float)
    data['z'] = z.astype(np.float)
    data['velocity'] = velocity.astype(np.float)
    data['intensity'] = intensity.astype(np.float)
    
    return data


# In[30]:


data = Get_data(filename)


# In[31]:


data


# In[32]:


data.describe()


# # Small tool functions

# In[54]:


def Normalize(x, x_min, x_max):
    return (x-x_min)/(x_max-x_min)


# In[ ]:





# # Plot data points

# In[128]:


def Get_colors(label_list):
    color_list = ['r', 'b', 'g', 'c', 'm', 'darkorange', 'deepskyblue', 'blueviolet', 'crimson', 'orangered', 'k']
    return list(map(lambda x: color_list[x], label_list))


# In[113]:


# alpha = 'none', 'intensity', 'velocity'
def Plot_data(ax, datalist, **kwargs):
    datalist = datalist.reset_index()

    if 'color' not in kwargs.keys():
        kwargs['color'] = ['k']*len(datalist)
    elif len(kwargs['color']) != len(datalist):
        kwargs['color'] = list(kwargs['color'])*len(datalist)

    if 'alpha' in kwargs.keys() and kwargs['alpha'] in ['intensity', 'velocity']:
        weight_min = np.min(np.abs(datalist[kwargs['alpha']]))
        weight_max = np.max(np.abs(datalist[kwargs['alpha']]))
    
        for i in range(len(datalist)):
            ax.scatter(datalist.x[i], datalist.y[i], datalist.z[i], color = kwargs['color'][i], alpha = Normalize(datalist.loc[i][kwargs['alpha']], weight_min, weight_max), marker = '.')
    else:
        for i in range(len(datalist)):
            ax.scatter(datalist.x[i], datalist.y[i], datalist.z[i], color = kwargs['color'][i], marker = '.')


# In[60]:


# Plot settings
def Plot_setting(ax):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
#     ax.set_xlim(-1, 5)
#     ax.set_ylim(-1, 2)
#     ax.set_zlim(-0.5, 1.5)


# # Choose frame window

# In[49]:


# How many data points in each frame
frame_point_count = []
for i in range(np.max(data.frame_num)+1):
    frame_point_count.append(len(data[data.frame_num==i]))

pd.Series(frame_point_count).describe()


# In[83]:


# How many data points in each 5-frame window
frame_point_count = []
WINDOW = 5
for i in range(int(np.max(data.frame_num)/WINDOW)):
    frame_point_count.append(len(data[(data.frame_num>=i*WINDOW)&(data.frame_num<(i+1)*WINDOW)]))

pd.Series(frame_point_count).describe()


# In[114]:


fig = plt.figure()
ax = fig.gca(projection = '3d')
Plot_setting(ax)

Plot_data(ax, data[data.frame_num<5], alpha='intensity', color='r')
Plot_data(ax, data[(data.frame_num<5)&(data.velocity==0)], color='k')

plt.show()


# In[115]:


fig = plt.figure()
ax = fig.gca(projection = '3d')
Plot_setting(ax)

Plot_data(ax, data[data.frame_num<5])

plt.show()


# # DBSCAN clustering

# In[165]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


# In[200]:


def Model_cluster(datalist, model, sample_weight=False, show_plot=True, return_cluster=False):
    if sample_weight:
        sample_weight = MinMaxScaler().fit_transform(np.asarray(datalist.intensity).reshape(-1,1)).reshape(1,-1)[0]
    else:
        sample_weight = None
    clustering = model.fit(np.asarray(datalist.iloc[:,1:4]), sample_weight = sample_weight)
    
    if show_plot:
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        Plot_setting(ax)
        
        Plot_data(ax, datalist, color = Get_colors(clustering.labels_))
        
        plt.show()
    
    if return_cluster:
        return clustering


# ## Compare with and without sample_weight

# In[202]:


# Clustering frame_num = 0~4
dbscan = DBSCAN(eps=0.4, min_samples=10)
datalist = data[data.frame_num<5]

Model_cluster(datalist, dbscan)
Model_cluster(datalist, dbscan, sample_weight=True)


# In[203]:


# Clustering frame_num = 5~9
dbscan = DBSCAN(eps=0.4, min_samples=10)
datalist = data[(data.frame_num<10)&(data.frame_num>=5)]

Model_cluster(datalist, dbscan)
Model_cluster(datalist, dbscan, sample_weight=True)


# In[204]:


# Clustering frame_num = 10~14
dbscan = DBSCAN(eps=0.4, min_samples=10)
datalist = data[(data.frame_num<15)&(data.frame_num>=10)]

Model_cluster(datalist, dbscan)
Model_cluster(datalist, dbscan, sample_weight=True)


# In[205]:


# Clustering frame_num = 15~19
dbscan = DBSCAN(eps=0.4, min_samples=10)
datalist = data[(data.frame_num<20)&(data.frame_num>=15)]

Model_cluster(datalist, dbscan)
Model_cluster(datalist, dbscan, sample_weight=True)


# ## Change DBSCAN eps, min_samples [TODO]

# In[ ]:





# # Draw boundary from cluster

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




