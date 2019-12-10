import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib, time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from sklearn.cluster import DBSCAN


# Open File
with open('sample_walk_1.txt') as f:
    lines = f.readlines()

frame_num_count = -1
frame_num = []
x = []
y = []
z = []
velocity = []
intensity = []

wordlist = []

for x1 in lines:
	for word in x1.split():
		wordlist.append(word)


length1 = len(wordlist)

for i in range(0,length1):
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

final = np.zeros([len(frame_num), 6])

i = 0

while i < len(frame_num):
	final[i,0] = int(frame_num[i])
	final[i,1] = x[i]
	final[i,2] = y[i]
	final[i,3] = z[i]
	final[i,4] = velocity[i]
	final[i,5] = intensity[i]
	i = i + 1


np.savetxt("read_output.csv", final, delimiter=",")


# x_max = max(final[:,1])
# y_max = max(final[:,2])
# z_max = max(final[:,3])
# x_min = min(final[:,1])
# y_min = min(final[:,2])
# z_min = min(final[:,3])

x_max = 4.5
y_max = 2
z_max = 1
x_min = 0.5
y_min = -1
z_min = -1.5

number_of_frames = len(set(frame_num))
max_frame = max(final[:,0])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def animate(i):
	x = []
	y = []
	z = []
	cloud = []
	result = []
	db_clusters_x = []
	db_clusters_y = []
	for index in range(len(frame_num)):
		if final[index, 0] == i:
			x.append(final[index, 1])
			y.append(final[index, 2])
			z.append(final[index, 3])
	for i1 in range(len(x)):
		result.append( [x[i1], y[i1]] )
	clustering = DBSCAN(eps=0.1, min_samples=5).fit(result)
	clusters = clustering.labels_

	for i2 in range(len(result)):
		for i3 in range(len(clusters)):
			if clusters[i3] == 0:
				db_clusters_x.append(x[i3])
				db_clusters_y.append(y[i3])

	border_x_max = max(db_clusters_x)
	border_y_max = max(db_clusters_y)
	border_x_min = min(db_clusters_x)
	border_y_min = min(db_clusters_y)
	border_x = [border_x_max, border_x_max, border_x_min, border_x_min]
	border_y = [border_y_max, border_y_min, border_y_max, border_y_min]

	ax.clear()
	ax.set_xlim3d(x_min, x_max)
	ax.set_ylim3d(y_min, y_max)
	ax.set_zlim3d(z_min, z_max)
	ax.scatter(x, y, z, c='r', marker='o')
	ax.plot(border_x, border_y)
	if i == max_frame:
		plt.close('all')
ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()

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