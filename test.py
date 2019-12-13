import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import math
import scipy.stats as sta
from scipy.optimize import curve_fit

# Open file
with open("sample_walk_1_frame1.txt") as f:
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
            data[frame_num[i]] = []
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


# To find x_min, x_max
def find_x_bounds(x_data, pillar = 32, debug = False):
    if debug:
        ver, hor, patch = plt.hist(x_data, bins = pillar, density = True)
    else:
        ver, hor = np.histogram(x_data, bins = pillar, density = True)
    mean = np.mean(x_data)
    std = np.std(x_data)
    fit = sta.norm.pdf(hor, mean, std)
    fwhm1 = (max(fit)+fit[0])/2
    fwhm2 = (max(fit)+fit[-1])/2

    x_min = min(x_data)
    x_max = max(x_data)
    for i in range(len(fit)-1):
        if fit[i] < fwhm1 <= fit[i+1]:
            x_min = hor[i+1]
        elif fit[i] >= fwhm2 > fit[i+1]:
            x_max = hor[i]

    if debug:
        plt.plot(hor, fit)
        plt.axvline(x=x_min)
        plt.axvline(x=x_max)
        print("x_min=" + str(x_min) + "\tx_max=" + str(x_max))

    return x_min, x_max


# To find y_min, y_max
def find_y_bounds(y_data, pillar = 32, debug = False):
    if debug:
        ver, hor, patch = plt.hist(y_data, bins = pillar, density = True)
    else:
        ver, hor = np.histogram(y_data, bins = pillar, density = True)
    mean = np.mean(y_data)
    std = np.std(y_data)

    y_min = min(y_data)
    y_max = max(y_data)
    y_min = max(y_min, mean-5*std)
    y_max = min(y_max, mean+5*std)

    if debug:
        fit = sta.norm.pdf(hor, mean, std)
        plt.plot(hor, fit)
        plt.axvline(x=y_min)
        plt.axvline(x=y_max)
        print("y_min=" + str(y_min) + "\ty_max=" + str(y_max))

    return y_min, y_max


# To find z_min, z_max
def find_z_bounds(z_data, pillar = 32, debug = False):
    if debug:
        ver, hor, patch = plt.hist(z_data, bins = pillar, density = True)
    else:
        ver, hor = np.histogram(z_data, bins = pillar, density = True)
    fit, cov = curve_fit(Gaussian2, hor[:-1], ver, p0=[1,1,0,1,0.5,0.5])

    z_min = min(z_data)
    z_max = max(z_data)
    z_min = max(z_min, fit[2]-5*fit[4])
    z_max = min(z_max, fit[2]+3*fit[4])  # Gap between two Gaussians

    if debug:
        plt.plot(hor, Gaussian2(hor, *fit))
        plt.axvline(x=z_min)
        plt.axvline(x=z_max)
        print("z_min=" + str(z_min) + "\tz_max=" + str(z_max))
        print(fit[2], fit[4])

    return z_min, z_max


def sort_data(startFrame, endFrame, clean = True):
    i_prev = 0
    for i in range(len(frame_num)):
        if frame_num[i] == startFrame - 1:
            i_prev = i
            continue
        if frame_num[i] == endFrame:
            i_curr = i
            break

    print("startFrame="+str(startFrame)+"\tendFrame="+str(endFrame)+"\ti_prev="+str(i_prev)+"\ti_curr="+str(i_curr))

    # Gaussian fit, similar to the method of voxels
    pillar = 32
    x_min, x_max = find_x_bounds(x[i_prev:i_curr], pillar)
    y_min, y_max = find_y_bounds(y[i_prev:i_curr], pillar)
    z_min, z_max = find_z_bounds(z[i_prev:i_curr], pillar)

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
    

# Sorting order matters, x based on sorted z, y based on sorted x and z (z -> x -> y, because z most effective according to the preprocess, x being the next, y the least)
# Drawbacks: two more loops, calculation heavy (probably)
def sort_data2(startFrame, endFrame, clean = True):
    i_prev = 0
    for i in range(len(frame_num)):
        if frame_num[i] == startFrame - 1:
            i_prev = i
            continue
        if frame_num[i] == endFrame:
            i_curr = i
            break

    print("startFrame="+str(startFrame)+"\tendFrame="+str(endFrame)+"\ti_prev="+str(i_prev)+"\ti_curr="+str(i_curr))

    # Gaussian fit, similar to the method of voxels
    pillar = 32

    # For z_min, z_max
    z_min, z_max = find_z_bounds(z[i_prev:i_curr], pillar)

    # For x_min, x_max
    i_list_x = []
    for i in range(i_prev, i_curr):
        if z_min <= z[i] <= z_max:
            i_list_x.append(i)
    if i_list_x == []:
        return []
    else:
        x_min, x_max = find_x_bounds(x[i_list_x], pillar)

    # For y_min, y_max
    i_list_y = []
    for i in i_list_x:
        if x_min <= x[i] <= x_max:
            i_list_y.append(i)
    if i_list_y == []:
        return []
    else:
        y_min, y_max = find_y_bounds(y[i_list_y], pillar)

    # print(x_min, x_max, y_min, y_max, z_min, z_max)

    i_list = []
    
    if clean:
        for i in i_list_y:
            if y_min <= y[i] <= y_max:
                i_list.append(i)
    else:
        for i in range(i_prev, i_curr):
            i_list.append(i)

    # print("i_list = " + str(i_list))
    return i_list


def plot_data(ax, i_list, datalist, color = 'k'):
    intensity_min = min(intensity[i_list])
    intensity_max = max(intensity[i_list])

    for i in i_list:
        alpha = (intensity[i] - intensity_min)/(intensity_max - intensity_min)
        ax.scatter(datalist[i, 1], datalist[i, 2], datalist[i, 3], color = color, alpha = alpha, marker = '.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 2)
    ax.set_zlim(-1.5, 1.5)


# Shift and add
# Problem: if sorted data not clean enough, max velocity point may fall in noises, making the shifted data a mess -> Solution: use sort_data2 func, a lot cleaner
# Problem: data too sparse for 1 frame, may or may not have large distance change in 60 frames -> Solution: shift and add every 10 frame (?)
def shift_data(i_list):
    center = expect(i_list)

    final2 = np.zeros([len(frame_num), 6])

    # data2: To store frame_index with same frame_number in a dict
    data2 = dict()
    for i in i_list:
        if int(frame_num[i]) in data2:
            data2[frame_num[i]].append(i)
        else:
            data2[frame_num[i]] = []
            data2[frame_num[i]].append(i)

    # Shift each frame so that all velocity_max points for each frame in the same position as center of every 60 frames
    for each_frame in data2:
        vel_max_frame_index = data2[each_frame][np.argmax(velocity[data2[each_frame]])]
        for i in data2[each_frame]:
            final2[i, 0] = int(frame_num[i])
            final2[i, 1] = x[i] - (x[vel_max_frame_index] - center[0])
            final2[i, 2] = y[i] - (y[vel_max_frame_index] - center[1])
            final2[i, 3] = z[i] - (z[vel_max_frame_index] - center[2])
            final2[i, 4] = velocity[i]
            final2[i, 5] = intensity[i]

    return final2


# Velocity(/intensity)-weighted expected center position of the figure
def expect(i_list, ref = 'intensity'):
    if ref == 'velocity':
        weight = np.abs(final[i_list, 4])
    elif ref == 'intensity':
        weight = np.abs(final[i_list, 5])
    if np.sum(weight) != 0:
        x = np.dot(final[i_list, 1], weight)/np.sum(weight)
        y = np.dot(final[i_list, 2], weight)/np.sum(weight)
        z = np.dot(final[i_list, 3], weight)/np.sum(weight)
    else:
        x = np.average(final[i_list, 1])
        y = np.average(final[i_list, 2])
        z = np.average(final[i_list, 3])
    return [x, y, z]


def find_verts(i_list, datalist):
    x_min = min(datalist[i_list, 1])
    x_max = max(datalist[i_list, 1])
    y_min = min(datalist[i_list, 2])
    y_max = max(datalist[i_list, 2])
    z_min = min(datalist[i_list, 3])
    z_max = max(datalist[i_list, 3])
    velocity_min = min(datalist[i_list, 4])
    velocity_max = max(datalist[i_list, 4])
    intensity_min = min(datalist[i_list, 5])
    intensity_max = max(datalist[i_list, 5])

    center = expect(i_list)

    verts = [x_min, x_max, y_min, y_max, z_min, z_max, center]

    return verts


# Velocity(/intensity)-based weighted lower and upper bounds estimation
# Results: not good
def find_verts2(i_list, datalist):
    center = expect(i_list)

    # Initialize
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    z_min = 0
    z_max = 0
    velocity_min = 0    # To calculate total weight for lower bound
    velocity_max = 0    # To calculate total weight for upper bound
    intensity_min = 0
    intensity_max = 0

    for i in i_list:
        if datalist[i, 1] <= center[0]:
            x_min += datalist[i, 1] * datalist[i, 4]
            velocity_min += datalist[i, 4]
        else:
            x_max += datalist[i, 1] * datalist[i, 4]
            velocity_max += datalist[i, 4]
    x_min = x_min/velocity_min
    x_max = x_max/velocity_max

    velocity_min = 0
    velocity_max = 0
    for i in i_list:
        if datalist[i, 2] <= center[1]:
            y_min += datalist[i, 2] * datalist[i, 4]
            velocity_min += datalist[i, 4]
        else:
            y_max += datalist[i, 2] * datalist[i, 4]
            velocity_max += datalist[i, 4]
    y_min = y_min/velocity_min
    y_max = y_max/velocity_max

    velocity_min = 0
    velocity_max = 0
    for i in i_list:
        if datalist[i, 3] <= center[2]:
            z_min += datalist[i, 3] * datalist[i, 4]
            velocity_min += datalist[i, 4]
        else:
            z_max += datalist[i, 3] * datalist[i, 4]
            velocity_max += datalist[i, 4]
    z_min = z_min/velocity_min
    z_max = z_max/velocity_max

    verts = [x_min, x_max, y_min, y_max, z_min, z_max, center]

    return verts


def plot_cube(ax, verts, alpha = 1, color = 'r'):
    x_min, x_max, y_min, y_max, z_min, z_max, center = verts

    xx = [x_min, x_min, x_max, x_max, x_min]
    yy = [y_min, y_max, y_max, y_min, y_min]
    kwargs = {'alpha': alpha, 'color': color}
    ax.plot3D(xx, yy, [z_min]*5, **kwargs)
    ax.plot3D(xx, yy, [z_max]*5, **kwargs)
    ax.plot3D([x_min, x_min], [y_min, y_min], [z_min, z_max], **kwargs)
    ax.plot3D([x_min, x_min], [y_max, y_max], [z_min, z_max], **kwargs)
    ax.plot3D([x_max, x_max], [y_min, y_min], [z_min, z_max], **kwargs)
    ax.plot3D([x_max, x_max], [y_max, y_max], [z_min, z_max], **kwargs)

    # Estimate the center point of the figure
    # Use the point with max velocity as the estimated center
    # i_max = i_list[np.argmax(final[i_list, 4])]     
    # ax.scatter(x[i_max], y[i_max], z[i_max], color = 'r')
    # Use the velocity-weighted expected position as the estimated center
    ax.scatter(center[0], center[1], center[2], color = 'r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 2)
    ax.set_zlim(-1.5, 1.5)
    # plt.show()


def find_figure_frame(i_list, datalist):
    pass


get_data()
print(frame_num_count)

data, final = organized_data()

# fig = plt.figure()
# ax = fig.gca(projection = '3d')

# Average height from all frames
# min_ave=0
# max_ave=0
# max_abs=0
# min_abs=0
# count=0
# for i in range(0, int(frame_num_count/60)-1):
#     i_list = sort_data(i*60,i*60+60)
#     if i_list != []:
#         final2 = shift_data(i_list)
#         verts = find_verts(i_list, final2)
#         min_ave+=verts[0]
#         max_ave+=verts[1]
#         count+=1
#         min_abs=min(min_abs, verts[0])
#         max_abs=max(max_abs, verts[1])
#     del i_list
# print(min_ave/count, max_ave/count, (max_ave-min_ave)/count)
# print(min_abs, max_abs, max_abs-min_abs)


# Plot trajectory
# traj = np.zeros([int(frame_num_count/60)-1,3])
# for i in range(0, int(frame_num_count/60)-1):
#     i_list = sort_data2(i*60,i*60+60)
#     if i_list != []:
#         final2 = shift_data(i_list)
#         plot_data(ax, i_list, final2)
#         verts = find_verts(i_list, final2)
#         print(verts)
#         plot_cube(ax, verts,(i+1)/10)
#         traj[i, :]=expect(i_list)
#         del i_list
# ax.plot3D(traj[:,0],traj[:,1],traj[:,2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_xlim(-1, 5)
# ax.set_ylim(-1, 2)
# ax.set_zlim(-1.5, 1.5)
# print(traj)

# # For debug
fig = plt.figure()
ax = fig.gca(projection = '3d')
i_list = sort_data2(0,60)
# final2 = shift_data(i_list)
plot_data(ax, i_list, final)
verts = find_verts(i_list, final)
# print(expect(i_list))
plot_cube(ax, verts)

# fig = plt.figure()
# ax = fig.gca(projection = '3d')
# i_list = sort_data2(300,360)
# # final2 = shift_data(i_list)
# plot_data(ax, i_list, final)
# verts = find_verts(i_list, final)
# plot_cube(ax, verts)

fig = plt.figure()
ax = fig.gca(projection = '3d')
i_list = sort_data2(0,60, False)
plot_data(ax, i_list, final)

plt.show()


# For debug
# pillar = 32
# i_prev = 0
# i_curr = 1382
# find_x_bounds(x[i_prev:i_curr], pillar, True)
# plt.figure()
# find_y_bounds(y[i_prev:i_curr], pillar, True)
# plt.figure()
# find_z_bounds(z[i_prev:i_curr], pillar, True)
# plt.show()
