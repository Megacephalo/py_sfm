#!/usr/bin/env python
#!/usr/bin/python

import numpy as np
from numpy.linalg import svd, multi_dot

class Fundamental_matrix:
	def __init__(self, x1, x2):
		self._x1 = x1
		self._x2 = x2

	def estimate(self):
		'''
		Inputs:
			x1 - size (N x 2) matrix of feature points in image 1
			x2 - size (N x 2) matrix of the corresponding feature points (epipoints) in image 2
		Outputs:
			F - size (3 x 3) fundamental matrix with rank 2
		'''
		num_points = self._x1.shape[0]
		x1_pr = np.append(self._x1, np.ones((num_points, 1)), axis=1 )
		x2_pr = np.append(self._x2, np.ones((num_points, 1)), axis=1 )

		# 8-point algorithm
		A = np.empty( shape=(num_points, 9) )
		for i in range(num_points):
			A[i, :] = np.kron(x1_pr[i, :], x2_pr[i, :])

		# Solve for F
		__, __, V = svd(A, full_matrices=True)
		F1 = np.reshape( V[:,-1], (3, 3))
		UF, DF, VF = svd(F1)
		
		DF_p = np.array([ [DF[0] 	, 0 	, 0], 
						  [0 		, DF[1] , 0],
						  [0		,0		, 0] ])

		F = multi_dot([UF, DF_p, VF])

		return F

