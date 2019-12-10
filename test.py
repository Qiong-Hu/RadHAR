import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

# Open file
with open("sample_walk_1.txt") as f:
    lines = f.readlines()

frame_num_count = -1
frame_num = []
x = []
y = []
z = []
velocity = []
intensity = []

def get_data():
    global frame_num_count, frame_num
    global x, y, z, velocity, intensity
    wordlist = []
    for line in lines:
        for word in line.split():
            wordlist.append(word)

    for i in range(0, len(wordlist)):
        if wordlist[i] == "point_id:" and wordlist[i+1] == "0":
            frame_num_count += 1
        if wordlist[i] == "point_id:":
            frame_num.append(frame_num_count)
        if wordlist[i] == "x:":
            x.append(wordlist[i+1])
        if wordlist[i] == "y:":
            y.append(wordlist[i+1])
        if wordlist[i] == "z:":
            z.append(wordlist[i+1])
        if wordlist[i] == "velocity:":
            velocity.append(wordlist[i+1])
        if wordlist[i] == "intensity:":
            intensity.append(wordlist[i+1])

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    frame_num = np.asarray(frame_num)
    velocity = np.asarray(velocity)
    intensity = np.asarray(intensity)

    x = x.astype(np.float)
    y = y.astype(np.float)
    z = z.astype(np.float)
    velocity = velocity.astype(np.float)
    intensity = intensity.astype(np.float)
    frame_num = frame_num.astype(np.int)


def organized_data():
    data = dict()
    final = np.zeros([len(frame_num), 6])
    
    for i in range(len(frame_num)):
        if int(frame_num[i]) in data:
            data[frame_num[i]].append([x[i],y[i],z[i],velocity[i],intensity[i]])
        else:
            data[frame_num[i]]=[]
            data[frame_num[i]].append([x[i],y[i],z[i],velocity[i],intensity[i]])

        final[i,0] = int(frame_num[i])
        final[i,1] = x[i]
        final[i,2] = y[i]
        final[i,3] = z[i]
        final[i,4] = velocity[i]
        final[i,5] = intensity[i]

    return data, final

def plot_data(ax):
    intensity_max = max(final[:,5])
    intensity_min = min(final[:,5])
    for i in range(len(frame_num)):
        ax.scatter(x[i],y[i],z[i],color='k',alpha=(intensity[i]-intensity_min)/(intensity_max-intensity_min))
        # print(intensity[i]/intensity_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.show()



get_data()
data, final = organized_data()
# print(data)
# print(final)
print(frame_num_count)

fig = plt.figure()
ax = fig.gca(projection = '3d')

# plot_data(ax)
