import numpy as np
import math
import time
import scipy.stats as sta
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

W = 60    # Time frame window: 60 frames (= 2 seconds)

# Open file
with open("./data/sample_walk_1_part1.txt") as f:
# with open("./data/Train/jump/__jump_122s.txt") as f:
# with open("./data/Train/walk/___walk_191s.txt") as f:
# with open("./data/Train/squats/___squats_186s.txt") as f:
# with open("./data/Train/jump/___jump_180s.txt") as f:
# with open("./data/Train/jack/jacks_179s.txt") as f:
# with open("./data/Train/boxing/boxing_191s.txt") as f:
    lines = f.readlines()

frame_num_count = -1
frame_num = []
x = []
y = []
z = []
velocity = []
intensity = []


# Retrieve data from the given file to separate lists
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


# Organize data in multiple lists to a list "final"
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


# Define the function to fit distribution in Z dimension
def Gaussian2(x, *par):
    return par[0]*np.exp(-np.power(x-par[2], 2) / (2 * np.power(par[4], 2))) + par[1]*np.exp(-np.power(x-par[3], 2) / (2 * np.power(par[5], 2)))


# To find x_min, x_max
def find_x_bounds(x_data, pillar = 32, debug = False):
    kwargs = {'color': 'b'}
    if debug:
        plt.figure()
        ver, hor, patch = plt.hist(x_data, bins = pillar, density = True, **kwargs)
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
        kwargs = {'color': 'orange'}
        plt.plot(hor, fit, **kwargs)
        plt.xlabel("X")
        plt.axvline(x=x_min, **kwargs)
        plt.axvline(x=x_max, **kwargs)
        print("x_min=" + str(x_min) + "\tx_max=" + str(x_max))

    return x_min, x_max


# To find y_min, y_max
def find_y_bounds(y_data, pillar = 32, debug = False):
    kwargs = {'color': 'b'}
    if debug:
        plt.figure()
        ver, hor, patch = plt.hist(y_data, bins = pillar, density = True, **kwargs)
    else:
        ver, hor = np.histogram(y_data, bins = pillar, density = True)
    mean = np.mean(y_data)
    std = np.std(y_data)

    y_min = min(y_data)
    y_max = max(y_data)
    y_min = max(y_min, mean-3*std)
    y_max = min(y_max, mean+3*std)

    if debug:
        kwargs = {'color': 'orange'}
        fit = sta.norm.pdf(hor, mean, std)
        plt.plot(hor, fit, **kwargs)
        plt.xlabel("Y")
        plt.axvline(x=y_min, **kwargs)
        plt.axvline(x=y_max, **kwargs)
        print("y_min=" + str(y_min) + "\ty_max=" + str(y_max))

    return y_min, y_max


# To find z_min, z_max
def find_z_bounds(z_data, pillar = 32, debug = False, cov0 = 0.3):
    kwargs = {'color': 'b'}
    if debug:
        plt.figure()
        ver, hor, patch = plt.hist(z_data, bins = pillar, density = True, **kwargs)
    else:
        ver, hor = np.histogram(z_data, bins = pillar, density = True)
    fit, cov = curve_fit(Gaussian2, hor[:-1], ver, p0=[1,1,0,1,cov0,cov0])

    z_min = min(z_data)
    z_max = max(z_data)
    z_min = max(z_min, fit[2]-3*fit[4])
    z_max = min(z_max, fit[2]+3*fit[4])  # Gap between two Gaussians

    if debug:
        kwargs = {'color': 'orange'}
        plt.plot(hor, Gaussian2(hor, *fit), **kwargs)
        plt.xlabel("Z")
        plt.axvline(x=z_min, **kwargs)
        plt.axvline(x=z_max, **kwargs)
        print("z_min=" + str(z_min) + "\tz_max=" + str(z_max))
        print(fit[2], fit[4])

    return z_min, z_max


# TODO: Not tested yet, for future work
def train_z_cov(z_data, pillar = 32, cov0 = 0.3, step = 0.02):
    z_min = 0
    z_max = 0
    cov = cov0
    while z_min == 0 and z_max == 0:
        try:
            z_min, z_max = find_z_bounds(z_data, pillar = pillar, cov0 = cov)
        except:
            cov += step
    return z_min, z_max


# To filter out noises
def sort_data(startFrame, endFrame, clean = True):
    i_prev = 0
    for i in range(len(frame_num)):
        if frame_num[i] == startFrame - 1:
            i_prev = i
            continue
        if frame_num[i] == endFrame:
            i_curr = i
            break

    print("time="+str(startFrame/30)+"s\tstartFrame="+str(startFrame)+"\tendFrame="+str(endFrame)+"\ti_prev="+str(i_prev)+"\ti_curr="+str(i_curr))

    # Gaussian fit, similar to the method of voxels
    pillar = 32
    x_min, x_max = find_x_bounds(x[i_prev:i_curr], pillar)
    y_min, y_max = find_y_bounds(y[i_prev:i_curr], pillar)
    # z: curve_fit may not converge -> use iterative train_z_cov
    try:
        z_min, z_max = find_z_bounds(z[i_prev:i_curr], pillar)
    except:
        z_min, z_max = find_x_bounds(z[i_prev:i_curr], pillar)

    # print(x_min, x_max, y_min, y_max, z_min, z_max)

    i_list = []
    for i in range(i_prev, i_curr):
        if clean:
            if x_min <= x[i] <= x_max and z_min <= z[i] <= z_max:
                i_list.append(i)
        else:
            i_list.append(i)

    # print("i_list = " + str(i_list))
    return i_list
    

# Sorting order matters, x based on sorted z, y based on sorted x and z (z -> x -> y, because z most effective according to the preprocess, x being the next, y the least)
# Drawbacks: two more loops, calculation heavier (0.2s more)
def sort_data2(startFrame, endFrame, clean = True):
    i_prev = 0
    for i in range(len(frame_num)):
        if frame_num[i] == startFrame - 1:
            i_prev = i
            continue
        if frame_num[i] == endFrame:
            i_curr = i
            break

    print("time="+str(startFrame/30)+"s\tstartFrame="+str(startFrame)+"\tendFrame="+str(endFrame)+"\ti_prev="+str(i_prev)+"\ti_curr="+str(i_curr))

    # Gaussian fit, similar to the method of voxels
    pillar = 32

    # For z_min, z_max
    # z: curve_fit may not converge -> use iterative train_z_cov
    try:
        z_min, z_max = find_z_bounds(z[i_prev:i_curr], pillar)
    except:
        z_min, z_max = find_x_bounds(z[i_prev:i_curr], pillar)

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

    # TODO: use the expected point as a reference, instead of the max_velocity point
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


def normalize(x, x_min, x_max):
    return (x-x_min)/(x_max-x_min)


# Estimate the center position of the figure
def expect(i_list, index = 5):
    # index = 4: use "velocity" as weight; = 5: use "intensity" as weight

    # Use the point with max intensity as the estimated center (Obsolete)
    # i_max = i_list[np.argmax(final[i_list, 5])]
    # x = x[i_max]
    # y = y[i_max]
    # z = z[i_max]

    # Use the velocity(/intensity)-weighted expected position as the estimated center
    weight = np.abs(final[i_list, index])    

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


def plot_data(ax, i_list, datalist, color = 'k', index = 5):
    # index = 4: use "velocity" as weight; = 5: use "intensity" as weight
    weight_min = min(np.abs(datalist[i_list, index]))
    weight_max = max(np.abs(datalist[i_list, index]))

    for i in i_list:
        alpha = normalize(abs(datalist[i, index]), weight_min, weight_max)
        ax.scatter(datalist[i, 1], datalist[i, 2], datalist[i, 3], color = color, alpha = alpha, marker = '.')


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

    # Plot the center point of the figure
    # ax.scatter(center[0], center[1], center[2], color = 'r')


def plot_traj(ax, count):
    traj = []
    for i in range(0, count):
        i_list = sort_data2(i*W, (i+1)*W)
        if i_list != []:
            # final2 = shift_data(i_list)
            # plot_data(ax, i_list, final2)
            # verts = find_verts(i_list, final2)
            # plot_cube(ax, verts)
            traj.append(expect(i_list))
            del i_list

    traj = np.array(traj)
    ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], color = 'b')
    print("effective rate", len(traj), count, len(traj)/count)
    # print(traj)

    # Plot range of the traj
    x_min = min(traj[:, 0])
    x_max = max(traj[:, 0])
    y_min = min(traj[:, 1])
    y_max = max(traj[:, 1])
    z_min = min(traj[:, 2])
    z_max = max(traj[:, 2])
    verts = [x_min, x_max, y_min, y_max, z_min, z_max, 0]
    print(verts[:-1])
    plot_cube(ax, verts, 0.5)


# average figure size among all frames
def figure_size():
    dis_x = []
    dis_y = []
    dis_z = []

    for i in range(0, int(frame_num_count/W)):
        i_list = sort_data2(i*W, (i+1)*W)
        if len(i_list) > 1:
            final2 = shift_data(i_list)
            verts = find_verts(i_list, final2)
            # x_min, x_max, y_min, y_max, z_min, z_max, center = verts
            dis_x.append(verts[1]-verts[0])
            dis_y.append(verts[3]-verts[2])
            dis_z.append(verts[5]-verts[4])
            del i_list

    # TODO (need debug): why for jump (/etc) data, x(/y/z)_min=0
    print("x_min=" + str(min(dis_x)) + "m\tx_max=" + str(max(dis_x)) + "m\tx_ave=" + str(np.average(dis_x)) + "m")
    print("y_min=" + str(min(dis_y)) + "m\ty_max=" + str(max(dis_y)) + "m\ty_ave=" + str(np.average(dis_y)) + "m")
    print("z_min=" + str(min(dis_z)) + "m\tz_max=" + str(max(dis_z)) + "m\tz_ave=" + str(np.average(dis_z)) + "m")


# Simply use the figure height and perfect/average human body model
# Reference: http://humanproportions.com/
def plot_skeleton(ax, i_list, datalist, color = 'm'):
    x_min, x_max, y_min, y_max, z_min, z_max, center = verts
    x0, y0, z0 = center[:]

    body = []

    # TODO: 
    # if y_max - y_min >= x_max - x_min:
    #     # Posture in y-orientation
    # else:
    #     # Posture in x-orientation
    # (default in y-orientation for now)

    h = z_max - z_min
    body.append([x0, y0, h/2])                          # 0: spine base
    body.append([x0, y0, h*(7/8-1/24)])                 # 1: spine shoulder
    body.append([x0, y0, h*7/8])                        # 2: neck
    body.append([x0, y0, h-h/16])                       # 3: head
    body.append([x0, y0-h/8, body[1][2]])               # 4: left shoulder
    body.append([x0, body[4][1], body[1][2]-1.7*h/8])   # 5: left elbow
    body.append([x0, body[4][1], body[5][2]-1.8*h/8])   # 6: left hand
    body.append([x0, 2*y0-body[4][1], body[4][2]])      # 7: right shoulder
    body.append([x0, body[7][1], body[5][2]])           # 8: right elbow
    body.append([x0, body[7][1], body[6][2]])           # 9: right hand
    body.append([x0, y0-0.1*h, h/2])                    # 10: left hip
    body.append([x0, body[10][1], h/4])                 # 11: left knee
    body.append([x0, body[10][1], 0])                   # 12: left foot
    body.append([x0, 2*y0-body[10][1], body[10][2]])    # 13: right hip
    body.append([x0, body[13][1], body[11][2]])         # 14: right knee
    body.append([x0, body[13][1], body[12][2]])         # 15: right foot

    body = np.array(body)
    body = body + np.array([0, 0, z_min])

    # For plot reference
    # ax.plot3D([x1, x2], [y1, y2], [z1, z2], 'b')
    kwargs = {'color': color, 'marker': '.'}
    ax.plot3D(body[0:4,0], body[0:4,1], body[0:4,2], **kwargs)
    connects = []
    connects.append([1, 4, 5, 6])
    connects.append([1, 7, 8, 9])
    connects.append([0, 10, 11, 12])
    connects.append([0, 13, 14, 15])
    for connect in connects:
        ax.plot3D(body[connect,0], body[connect,1], body[connect,2], **kwargs)
    tips = [3, 6, 9, 12, 15]
    ax.scatter(body[tips,0], body[tips,1], body[tips,2], color = color, marker = 'o')
    plt.axis("square")


# TODO: Below are the second method to estimate figure posture/skeleton, need improvement
# To estimate torso body part
# Intensity-based weighted lower and upper bounds estimation
def find_torso(i_list, datalist):
    center = expect(i_list)

    # Initialize
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    z_min = 0
    z_max = 0

    weight_min = 0
    weight_max = 0
    index = 5       # index = 4: use "velocity" as weight; = 5: use "intensity" as weight

    for i in i_list:
        if datalist[i, 1] <= center[0]:
            x_min += datalist[i, 1] * abs(datalist[i, index])
            weight_min += abs(datalist[i, index])
        else:
            x_max += datalist[i, 1] * abs(datalist[i, index])
            weight_max += abs(datalist[i, index])
    x_min = x_min/weight_min
    x_max = x_max/weight_max

    weight_min = 0
    weight_max = 0
    for i in i_list:
        if datalist[i, 2] <= center[1]:
            y_min += datalist[i, 2] * abs(datalist[i, index])
            weight_min += abs(datalist[i, index])
        else:
            y_max += datalist[i, 2] * abs(datalist[i, index])
            weight_max += abs(datalist[i, index])
    y_min = y_min/weight_min
    y_max = y_max/weight_max

    weight_min = 0
    weight_max = 0
    for i in i_list:
        if datalist[i, 3] <= center[2]:
            z_min += datalist[i, 3] * abs(datalist[i, index])
            weight_min += abs(datalist[i, index])
        else:
            z_max += datalist[i, 3] * abs(datalist[i, index])
            weight_max += abs(datalist[i, index])
    z_min = z_min/weight_min
    z_max = z_max/weight_max

    verts = [x_min, x_max, y_min, y_max, z_min, z_max, center]

    return verts


# Use velocity for limbs estimation
def find_limbs(i_list, datalist, torso_verts):
    limbs = []

    argsort = np.argsort(datalist[i_list,:], axis = 0)
    z_min1 = argsort[0][3]  # foot 1
    z_min2 = argsort[1][3]  # foot 2
    z_max = argsort[-1][3]  # head
    y_min = argsort[0][2]   # hand 1
    y_max = argsort[-1][2]  # hand 2

    # Find tips
    limbs.append(i_list[z_max])         # 0: head
    if datalist[i_list[z_min1],2] <= datalist[i_list[z_min2],2]: # smaller y = left
        limbs.append(i_list[z_min1])    # 1: left foot
        limbs.append(i_list[z_min2])    # 2: right foot
    else:
        limbs.append(i_list[z_min2])    # 1: left foot
        limbs.append(i_list[z_min1])    # 2: right foot
    limbs.append(i_list[y_min])         # 3: left hand
    limbs.append(i_list[y_max])         # 4: right hand

    # Find joints
    y_small = []    # point_y <= center_y
    y_big = []      # point_y > center_y
    z_small = []    # point_z <= center_z
    for i in i_list:
        if datalist[i][2] <= torso_verts[2]:
            y_small.append(i)
        elif datalist[i][2] >= torso_verts[3]:
            y_big.append(i)
        if datalist[i][3] <= torso_verts[4]:
            z_small.append(i)
    # TODO: deal with situation when y_small, y_big, z_small == []

    argsort = np.argsort(datalist[y_small,:], axis = 0)
    y_elbow1 = argsort[int(len(y_small)/2)][2]
    argsort = np.argsort(datalist[y_big,:], axis = 0)
    y_elbow2 = argsort[int(len(y_big)/2)][2]
    argsort = np.argsort(datalist[z_small,:], axis = 0)
    z_knee1 = argsort[int(len(z_small)/2)][3]
    z_knee2 = argsort[int(len(z_small)/2)+1][3]

    if datalist[z_small[z_knee1],2] <= datalist[z_small[z_knee2],2]: # smaller y = left
        limbs.append(z_small[z_knee1])  # 5: left knee
        limbs.append(z_small[z_knee2])  # 6: right knee
    else:
        limbs.append(z_small[z_knee2])  # 5: left knee
        limbs.append(z_small[z_knee1])  # 6: right knee
    limbs.append(y_small[y_elbow1])     # 7: left elbow
    limbs.append(y_big[y_elbow2])       # 8: right elbow

    return limbs


# Plot torso and limbs
# Result: not good, easily affected by remainder noise
def plot_skeleton2(ax, i_list, datalist, color = 'm'):
    kwargs = {'color': color, 'marker': '.'}

    torso_verts = find_torso(i_list, datalist)
    x_min, x_max, y_min, y_max, z_min, z_max, center = torso_verts
    plot_cube(ax, torso_verts, color = color)
    
    limbs = find_limbs(i_list, datalist, torso_verts)

    body = []
    body = datalist[limbs,1:4]
    body = list(body)
    body.append([(x_min+x_max)/2, (y_min+y_max)/2, z_max])  # 9: spine shoulder
    body.append([(x_min+x_max)/2, y_min, z_max])    # 10: left shoulder
    body.append([(x_min+x_max)/2, y_max, z_max]) # 11: right shoulder
    body.append([(x_min+x_max)/2, y_min, z_min]) # 12: left hip
    body.append([(x_min+x_max)/2, y_max, z_min]) # 13: right hip
    body = np.array(body)
    # ax.scatter(body[:,0], body[:,1], body[:,2],color='b',alpha=1)

    connects = []
    connects.append([0,9])
    connects.append([3,7,10])
    connects.append([4,8,11])
    connects.append([1,5,12])
    connects.append([2,6,13])

    for connect in connects:
        ax.plot3D(body[connect,0], body[connect,1], body[connect,2], **kwargs)
    ax.scatter(body[:5,0], body[:5,1], body[:5,2], color = color, marker = 'o')
    # for limb in limbs:
        # ax.scatter(datalist[limb,1],datalist[limb,2],datalist[limb,3], color='b')
        # ax.plot3D([center[0],datalist[limb,1]],[center[1],datalist[limb,2]],[center[2],datalist[limb,3]], color='b')
    plt.axis("square")


# Plot settings, same for all plots
def ax_settings(ax):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim(-1, 5)
    # ax.set_ylim(-1, 2)
    # ax.set_zlim(-1.5, 1.5)


# For experiment
get_data()
print(frame_num_count)
data, final = organized_data()

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax_settings(ax)

plot_traj(ax, int(frame_num_count/W))
figure_size()

plt.show()
# fig.savefig("1.jpg", dpi = 300)

# For debug
# start=120
# end=180
# i_list = sort_data2(start,end,False)
# plot_data(ax, i_list, final)
# i_list = sort_data2(start,end)
# verts = find_verts(i_list, final)
# plot_cube(ax, verts)
# # final2 = shift_data(i_list)
# plot_skeleton(ax, i_list, final2)

# Plot figure in multi-frame
# for i in range(0,10):
#     i_list = sort_data2(60*i, 60*i+60)
#     final2 = shift_data(i_list)
#     # plot_data(ax, i_list, final2)
#     verts = find_verts(i_list, final2)
#     plot_cube(ax, verts, 0.5)

    # plot_skeleton(ax, i_list, final2)

# fig = plt.figure()
# ax = fig.gca(projection = '3d')
# i_list = sort_data2(300,360)
# # final2 = shift_data(i_list)
# plot_data(ax, i_list, final)
# verts = find_verts(i_list, final)
# plot_cube(ax, verts)

# fig = plt.figure()
# ax = fig.gca(projection = '3d')
# ax_settings(ax)
# i_list = sort_data(0,60, False)
# plot_data(ax, i_list, final)

# plt.show()
# fig.savefig("1.jpg", dpi = 300)


# For debug
# pillar = 32
# i_prev = 0
# i_curr = 1382
# find_x_bounds(x[i_prev:i_curr], pillar, True)
# find_y_bounds(y[i_prev:i_curr], pillar, True)
# find_z_bounds(z[i_prev:i_curr], pillar, True)
# plt.show()
