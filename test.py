import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import math
import scipy.stats as sta
from scipy.optimize import curve_fit

# Open file
with open("sample_walk_1_part1.txt") as f:
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


def Gaussian2(x, *par):
    return par[0]*np.exp(-np.power(x-par[2], 2) / (2 * np.power(par[4], 2))) + par[1]*np.exp(-np.power(x-par[3], 2) / (2 * np.power(par[5], 2)))


def sort_data(startFrame, endFrame, clean = True):
    i_prev = 0
    for i in range(len(frame_num)):
        if frame_num[i] == startFrame - 1:
            i_prev = i
            continue
        if frame_num[i] == endFrame:
            i_curr = i
            break

    # print("i_prev=" + str(i_prev) + "\ti_curr=" + str(i_curr))

    # Gaussian fit, similar to the method of voxels
    pillar = 32
    # For x_min, x_max
    ver, hor = np.histogram(x[i_prev:i_curr], bins = pillar, density = True)
    mean = np.mean(x[i_prev:i_curr])
    fit = sta.norm.pdf(hor, mean, np.std(x[i_prev:i_curr]))
    fwhm = (max(fit)+min(fit))/2
    x_min = min(x[i_prev:i_curr])
    x_max = max(x[i_prev:i_curr])
    for i in range(len(fit)-1):
        if fit[i] < fwhm <= fit[i+1]:
            x_min = hor[i+1]
        elif fit[i] >= fwhm > fit[i+1]:
            x_max = hor[i]

    # For y_min, y_max
    ver, hor = np.histogram(y[i_prev:i_curr], bins = pillar, density = True)
    mean = np.mean(y[i_prev:i_curr])
    std = np.std(y[i_prev:i_curr])
    y_min = min(y[i_prev:i_curr])
    y_max = max(y[i_prev:i_curr])
    # print(y_min, y_max)
    y_min = max([y_min, mean-5*std])
    y_max = min([y_max, mean+5*std])

    # For z_min, z_max
    ver, hor = np.histogram(z[i_prev:i_curr], bins = pillar, density = True)
    fit, cov = curve_fit(Gaussian2, hor[:-1], ver, p0=[1,1,0,1,0.5,0.5])
    # plt.plot(hor, Gaussian2(hor, *fit))
    z_min = min(z[i_prev:i_curr])
    z_max = max(z[i_prev:i_curr])
    z_min = fit[2]-5*fit[4]
    z_max = (fit[2]+3*fit[4]+fit[3]-3*fit[5])/2  # Gap between two Gaussians

    # print(x_min, x_max, y_min, y_max, z_min, z_max)

    i_list = []
    for i in range(i_prev, i_curr):
        if clean:
            if x_min <= x[i] <= x_max and y_min <= y[i] <= y_max and z_min <= z[i] <= z_max:
                i_list.append(i)
        else:
            i_list.append(i)

    # print("i_list = " + str(i_list))
    return i_list
    

def plot_data(ax, i_list, color = 'k'):
    intensity_min = min(final[i_list, 5])
    intensity_max = max(final[i_list, 5])

    for i in i_list:
        ax.scatter(x[i], y[i], z[i], color = color, alpha = (intensity[i] - intensity_min)/(intensity_max - intensity_min), marker = '.')
        # ax.scatter(x[i] - x[i_max], y[i] - y[i_max], z[i] - z[i_max], color = color, alpha = (intensity[i] - intensity_min)/(intensity_max - intensity_min))

    # Frame number of the point with the max velocity (and not on the walls or ceiling)
    i_max = i_list[np.argmax(final[i_list, 4])]
    i = i_max
    ax.scatter(x[i], y[i], z[i], color = 'r', alpha = (intensity[i] - intensity_min)/(intensity_max - intensity_min))
    # print(i_max, x[i], y[i], z[i], (intensity[i] - intensity_min)/(intensity_max - intensity_min))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 2)
    ax.set_zlim(-1.5, 1.5)
    # plt.show()


# Shift and add
def plot_data2(ax, i_list, color = 'k'):
    intensity_min = min(final[i_list, 5])
    intensity_max = max(final[i_list, 5])

    # Frame number of the point with the max velocity (and not on the walls or ceiling)
    i_max = i_list[np.argmax(final[i_list, 4])]

    switch = np.zeros(len(i_list))
    data_new = dict()
    for i in i_list:
        if int(frame_num[i]) in data_new:
            data_new[frame_num[i]].append(i)
        else:
            data_new[frame_num[i]]=[]
            data_new[frame_num[i]].append(i)

    for each_frame in data_new:
        vel_max_frame_index = data_new[each_frame][np.argmax(velocity[data_new[each_frame]])]
        for i in data_new[each_frame]:
            ax.scatter(x[i]-(x[vel_max_frame_index]-x[i_max]), y[i]-(y[vel_max_frame_index]-y[i_max]), z[i]-(z[vel_max_frame_index]-z[i_max]), color = color, alpha = (intensity[i] - intensity_min)/(intensity_max - intensity_min))
    i = i_max
    ax.scatter(x[i], y[i], z[i], color = 'r', alpha = (intensity[i] - intensity_min)/(intensity_max - intensity_min))
    print(i_max, x[i], y[i], z[i], (intensity[i] - intensity_min)/(intensity_max - intensity_min))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 2)
    ax.set_zlim(-1.5, 1.5)
    # plt.show()


def plot_cube(ax, i_list, alpha = 1, color = 'r'):
    x_min = min(final[i_list, 1])
    x_max = max(final[i_list, 1])
    y_min = min(final[i_list, 2])
    y_max = max(final[i_list, 2])
    z_min = min(final[i_list, 3])
    z_max = max(final[i_list, 3])
    velocity_min = min(final[i_list, 4])
    velocity_max = max(final[i_list, 4])
    intensity_min = min(final[i_list, 5])
    intensity_max = max(final[i_list, 5])

    xx = [x_min, x_min, x_max, x_max, x_min]
    yy = [y_min, y_max, y_max, y_min, y_min]
    kwargs = {'alpha': alpha, 'color': color}
    ax.plot3D(xx, yy, [z_min]*5, **kwargs)
    ax.plot3D(xx, yy, [z_max]*5, **kwargs)
    ax.plot3D([x_min, x_min], [y_min, y_min], [z_min, z_max], **kwargs)
    ax.plot3D([x_min, x_min], [y_max, y_max], [z_min, z_max], **kwargs)
    ax.plot3D([x_max, x_max], [y_min, y_min], [z_min, z_max], **kwargs)
    ax.plot3D([x_max, x_max], [y_max, y_max], [z_min, z_max], **kwargs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 2)
    ax.set_zlim(-1.5, 1.5)
    # plt.show()


get_data()
print(frame_num_count)

data, final = organized_data()

fig = plt.figure()
ax = fig.gca(projection = '3d')
i_list = sort_data(0, 60)
# plot_cube(ax, i_list, color='y')
plot_data(ax, i_list)

plt.show()



