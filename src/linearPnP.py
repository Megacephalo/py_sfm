#!/usr/bin/python
#!/usr/bin/env python

import numpy as np
from numpy.linalg import solve, svd, det

class LinearPnP:
	'''
	Getting camera pose from 2D-3D correspondences
	Inputs:
		X - size (N x 3) matrix of 3D points
		x - size (N x 2) matrix of 2D points associated with their 3D counterparts
		K - size (3 x 3) camera calibration (intrinsics) matrix
	Outputs:
		C - size (3 x 1) pose translation
		R - size (3 x 1) pose rotation 

	IMPORTANT NOTE: While theoretically you can use the x directly when solving
	for the P = [R t] matrix then use the K matrix to correct the error, this is
	more numeically unstable, and thus it is better to calibrate the x values
	before the computation of P then extract R and t directly
	'''
	def __init__(self, X, x, K):
		self._X = X
		self._x = x
		self._K = K

		if not self.areValidArgs():
			raise Exception('The input arguments are not valid. Please check again.')

	def estimate(self):
		if not self.areValidArgs():
			raise Exception('The input arguments are not valid. Please check again.')

		# for the 2D points
		num_points = self._x.shape[0]

		# x1 = [x 1] has size N x 3
		x1 = np.concatenate((self._x, np.ones(shape=(num_points, 1))), axis=1)
		# xc = K \ x1' i.e. Solve xc for K * xc = x1^T using least squares
		# But actually, it should be more like x_c = K^{-1} x
		xc = solve(self._K, x1.transpose())
		# xc' has size N x 3
		xc = xc.transpose()

		# onto 3D points
		num_3d_pts = self._X.shape[0]
		# X1 = [X ones(num_3d_pts, 1)]  X1 has size N x 4
		X1 = np.concatenate((self._X, np.ones(shape=(num_points, 1))), axis=1)

		# zeros1x4 has size N x 4 
		zeros1x4 = np.zeros((num_3d_pts, 4))
		
		# u and v both have size N x 1
		u = xc[:, 0].reshape(num_3d_pts, 1)
		v = xc[:, 1].reshape(num_3d_pts, 1)

		# A has size 3N x 12
		A = np.concatenate(( np.concatenate((	zeros1x4			,	-X1 			 , np.multiply(v, X1)	)	, axis=1),
				  			 np.concatenate((	X1 					, zeros1x4 			 , -np.multiply(u, X1)	)	, axis=1),
					  		 np.concatenate((	-np.multiply(v, X1) , np.multiply(u, X1) , zeros1x4				)	, axis=1) ), axis=0)

		# Solve SVD for A
		# V has size 12 x 12
		__, __, V = svd(A)

		# Compute projection matrix P
		# P has size 3 x 4
		P = V[:, -1].reshape(3, 4)

		# Once P is found, use K to find  [R t]. The [R t] would have size 3 x 4
		# The first three columns belong to R while the last belongs to t.
		R1 = P[:, :-1] # all but the last column. Size (3 x 3)
		t1 = P[:, -1] # the last column. Size (3 x 1)

		U, D, V = svd(R1)
		# R = U * V
		R = np.matmul(U, V)

		detR = det(R)
		thresh = 0.001
		t = np.empty((3, 1))
		sigma1 = D[0]
		if abs(detR - 1.) < thresh:
			t = t1 / sigma1
		elif abs(detR + 1.) < thresh:
			R *= -1
			t = -t1 / sigma1
		else:
			raise Exception('Determinant of R is not 1 not -1. Please check all previous procedure and make sure everything is correct.')

		# C = -R * t
		C = -np.matmul(R.transpose(), t)
		C = C.reshape(3, 1)

		return C, R


	def areValidArgs(self):
		if self._X is None:
			return False

		if self._x is None:
			return False

		if self._K is None:
			return False

		X_r, X_c = self._X.shape
		x_r, x_c = self._x.shape
		if X_r != x_r and X_c != x_c:
			return False

		K_r, K_c = self._K.shape
		if K_r != 3 and K_c != 3:
			return False

		return True