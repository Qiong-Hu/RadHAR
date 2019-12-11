import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

# Open file
with open("C:\Users\qiong\Downloads\sample_walk_1_part1.txt") as f:
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
            data[frame_num[i]].append([x[i], y[i], z[i], velocity[i], intensity[i]])
        else:
            data[frame_num[i]]=[]
            data[frame_num[i]].append([x[i], y[i], z[i], velocity[i], intensity[i]])

        final[i, 0] = int(frame_num[i])
        final[i, 1] = x[i]
        final[i, 2] = y[i]
        final[i, 3] = z[i]
        final[i, 4] = velocity[i]
        final[i, 5] = intensity[i]

    return data, final


def sort_data(startFrame, endFrame, threshold = [0.3, 0.5, 0.1]):
    i_prev = 0
    for i in range(len(frame_num)):
        if frame_num[i] == startFrame - 1:
            i_prev = i
            continue
        if frame_num[i] == endFrame:
            i_curr = i
            break

    # print("i_prev = " + str(i_prev))
    # print("i_curr = " + str(i_curr))

    x_min = min(final[i_prev: i_curr, 1])
    x_max = max(final[i_prev: i_curr, 1])
    y_min = min(final[i_prev: i_curr, 2])
    y_max = max(final[i_prev: i_curr, 2])
    z_min = min(final[i_prev: i_curr, 3])
    z_max = max(final[i_prev: i_curr, 3])

    i_list = []
    for i in range(i_prev, i_curr):
        if x_min + threshold[0] <= x[i] <= x_max - threshold[0] and y_min + threshold[1] <= y[i] and z_min + threshold[2] <= z[i] <= z_max - threshold[2]:
            i_list.append(i)

    # print("i_list = " + str(i_list))

    velocity_min = min(final[i_list, 4])
    velocity_max = max(final[i_list, 4])
    intensity_min = min(final[i_list, 5])
    intensity_max = max(final[i_list, 5])

    # Frame number of the point with the max intensity or velocity (and not on the walls or ceiling)
    i_max = i_list[np.argmax(final[i_list, 4])]
    # i_max = i_list[np.argmax(final[i_list, 5])]

    return i_list, i_max
    

def plot_data(ax, i_list, i_max):
    intensity_min = min(final[i_list, 5])
    intensity_max = max(final[i_list, 5])

    for i in i_list:
        ax.scatter(x[i], y[i], z[i], color = 'k', alpha = (intensity[i] - intensity_min)/(intensity_max - intensity_min))

    i = i_max
    ax.scatter(x[i], y[i], z[i], color = 'r', alpha = (intensity[i] - intensity_min)/(intensity_max - intensity_min))
    print(i_max, x[i], y[i], z[i], (intensity[i] - intensity_min)/(intensity_max - intensity_min))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()



get_data()
print(frame_num_count)

data, final = organized_data()
i_list, i_max = sort_data(0, 60)

fig = plt.figure()
ax = fig.gca(projection = '3d')

plot_data(ax, i_list, i_max)


