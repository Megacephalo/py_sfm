#!/usr/bin/env python
#!/usr/bin/python

import numpy as np
from numpy.linalg import svd

from utilities import *

class Linear_triangulation:
	'''
	Find 3D positions of a point from its correspondences in two frames
	using relative position of one camera from another
	
	Inputs:
		K - size (3 x 3) Camera matrix
		C1 - size (3 x 1) translation of the first camera pose
		R1 - size (3 x 3) rotation of the first cmaera pose
		C2 - size (3 x 1) translation of the second camera
		R2 - size (3 x 3) rotation of the second camera pose
		x1 - size (N x 2) matrix of feature points in image 1
		x2 - size (N x 2) matrix of feature points in image 2
	Outputs:
		X - size (N x 3) matrix whose rows store the triangulated 3D points
	'''
	def __init__(self, K, C1, R1, C2, R2, x1, x2):
		self._K = K
		self._C1 = C1
		self._R1 = R1
		self._C2 = C2
		self._R2 = R2
		self._x1 = x1
		self._x2 = x2

	def estimate(self):
		# P1 = K * [R1 -R1*C1]  P1 has size 3 x 4
		T1 = np.matmul(-self._R1, self._C1)
		P1 = np.matmul( self._K, np.hstack( (self._R1, T1) ) )

		# P2 = K * [R2 -R2*C2]  P2 has size 3 x 4
		T2 = np.matmul(-self._R2, self._C2)
		P2 = np.matmul( self._K, np.hstack( (self._R2, T2) ) )

		# Take a correspondence pair from both frames. Perform triangulation
		num_points = self._x1.shape[0]
		X = np.empty(shape=(num_points, 3))
		idx = 0
		for corr in np.c_[self._x1, self._x2]:
			# corr has the form of [u1 v1 u2 v2]
			x1 = np.array([ corr[0], corr[1], 1])
			x2 = np.array([ corr[2], corr[3], 1])

			skew1 = vec2skew(x1)
			skew2 = vec2skew(x2)

			# A = [skew1 * P1 ; skew2 * P2] A has size = 6 x 4
			a1 = np.matmul(skew1, P1)
			a2 = np.matmul(skew2, P2)
			A = np.vstack((a1, a2))

			__, __, V = svd(A)

			X1 = V[:, -1] / V[-1, -1]
			X1 = X1[:-1]

			# X = np.append(X, X1, axis=0)
			X[idx] = X1
			idx += 1

		return X