# RadHAR
Human Activity Recognition from Point Clouds Generated through a Millimeter-wave Radar

An extension work of https://github.com/nesl/RadHAR, the previous published work can be found on https://doi.org/10.1145/3349624.3356768

This extended work is mainly contributed by Qiong Hu, supported by Akash Deep Singh and Ziqi Wang.

- Objective:
	- Filter out noise signals from Radar point clouds
	- Retrieve and generate human figures
	- Recognize and classify human motions
- Methods: 
	- Extract the characteristics of data distribution individually on three dimensions
	- Use statistic functions to fit the target signal area in the data distribution histogram
	- Filter out noise signals
	- Use shift-and-add to obtain target figure information
	- Algorithm can be found in [filter.py](https://github.com/Qiong-Hu/RadHAR/blob/master/filter.py)
- Results/Applications:
	- The filtering algorithm can effectively filter out environmental noises, estimate the human figure center and generate human cuboid representation every two seconds
	- Using the center points from all the datas can generate human motion trajectory
	- Using the cuboid representation can estimate human figure size and skeleton representation
	- The filtering algorithm is generated from one single dataset from one walking experiment, and is applicable for other walking experiments and more motions without having pre-knowledge or changing parameters
	- Resulting images can be found in [img/](https://github.com/Qiong-Hu/RadHAR/tree/master/img)
	- Resulting datas of trajectory estimation and figure size estimation can be found in [results.yaml](https://github.com/Qiong-Hu/RadHAR/blob/master/results.yaml)
- Future work:
	- About Methods:
		- Fitting function for data distribution in Z dimension may converge, which may be replaced by a learning algorithm
		- The shift-and-add function still need debug about which point in every frame to align to the center from 60 frames
		- The figure skeleton is based on fixed human body proportion for now, which may be replaced by information extracted from the filtered data points
	- About Applications:
		- The figure size estimation is not accurate enough yet
		- When the algorithm is applied to unfamiliar dataset, there would still be several noises that are not completely filtered out
		- The final objective would be to extract accurate human posture and recognize/classify human motions

