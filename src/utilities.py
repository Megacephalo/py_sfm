#!/usr/bin/env python
#!/usr/bin/python

import numpy as np
from geometry_msgs.msg import Quaternion
from math import sqrt

def vec2skew(vector):
	'''
	Input:
		vector - size(3 x 1) The vectorized pose in the form of np.array([x, y, 1])
	Output:
		X - size(3 x 3) The skew symmetric matrix equivalent of the vector
	'''
	if vector is None:
		raise Exception('The given vector is empty')
	if vector.shape[0] != 3:
		raise Exception('The given vector does not comply with the 3D dimension.')

	u12 = -1 * vector[2]
	u13 = vector[1]
	u21 = vector[2]
	u23 = -1 * vector[0]
	u31 = -1 * vector[1]
	u32 = vector[0]
	return np.array([[0, u12, u13],
					[u21, 0, u23],
					[u31, u32, 0]])

def rotationMatrix_2_quaternion(R):
	'''
	Input:
		R - size (3 x 3) rotation matrix of form
			r00  r01  r02
			r10  r11  r12
			r20  r21  r22
	Output:
		q - Quaternion message
	A shout-out to https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
	'''
	if R is None:
		raise Exception('The given rotation matrix does not exist. Please check input.')

	row, col = R.shape
	if row != col:
		raise Exception('The given rotation matrix is not valid. Please check input.')
	if row != 3 or col != 3:
		raise Exception('The given rotation matrix does not has size 3 x 3. Please check input.')

	trace = R[0, 0] + R[1, 1] + R[2, 2]
	qw, qx, qy, qz = 0, 0, 0, 0
	if trace > 0:
		S = 2. * sqrt(trace + 1.)
		qw = 0.25 * S
		qx = (R[2, 1] - R[1, 2]) / S
		qy = (R[0, 2] - R[2, 0]) / S
		qz = (R[1, 0] - R[0, 1]) / S
	elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
		S = sqrt(1. + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * qx
		qw = (R[2, 1] - R[1, 2]) / S
		qx = 0.25 * S
		qy = (R[0, 1] + R[1, 0]) / S
		qz = (R[0, 2] + R[2, 0]) / S
	elif R[1, 1] > R[2, 2]:
		S = sqrt(1. + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * qy
		qw = (R[0, 2] - R[2, 0]) / S
		qx = (R[0, 1] + R[1, 0]) / S
		qy = 0.25 * S
		qz = (R[1, 2] + R[2, 1]) / S
	else:
		S = sqrt(1. + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * qz
		qw = (R[1, 0] - R[0, 1]) / S
		qx = (R[0, 2] + R[2, 0]) / S
		qy = (R[1, 2] + R[2, 1]) / S
		qz = 0.25 * S

	return Quaternion(qx, qy, qz, qw)