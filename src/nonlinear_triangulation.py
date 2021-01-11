#!/usr/bin/env python
#!/usr/bin/python

import numpy as np
from numpy.linalg import multi_dot, inv

class NonLinear_Triangulation:
	'''
	Refine the camera poses to get a better estimate on the 3D point poses.
	Inputs:
		K 	- size (3 x 3) camera calibration (intrinsic) matrix
		C1 	- size (3 x 1) camera pose in frame 1
		R1 	- size (3 x 3) camera rotation matrix in frame 1
		C2 	- size (3 x 1) camera pose in frame 2
		R2 	- size (3 x 3) camera rotation matrix in frame 2
		C3 	- size (3 x 1) camera pose in frame 3
		R3 	- size (3 x 3) camera rotation matrix in frame 3
		x1 	- size (N x 2) 2D feature points in frame 1
		x2 	- size (N x 2) 2D feature points in frame 2
		x3 	- size (N x 2) 2D feature points in frame 3
		raw_X - size (N x 3) original 3D points that need optimizing
	Outputs:
		X 	- size (N x 3) matrix of refined 3D point poses
	'''
	def __init__(self, K, C1, R1, C2, R2, C3, R3, x1, x2, x3, raw_X):
		self._K = K
		self._C1 = C1
		self._R1 = R1
		self._C2 = C2
		self._R2 = R2
		self._C3 = C3
		self._R3 = R3
		self._x1 = x1
		self._x2 = x2
		self._x3 = x3
		self._X0 = raw_X

	def correct(self):
		num_points = self._X0.shape[0]
		if num_points is None or num_points == 0:
			raise Exception('There is no given point to do nonlinear triangulation. Check the inputs.')

		X = np.empty((num_points, 3))
		for idx, point in enumerate( np.c_[self._x1, self._x2, self._x3, self._X0] ):
			# point contains [u_pt1 v_pt1 u_pt2 v_pt2 u_pt3 v_pt3 x_X0 y_X0 z_x0]
			x1 = np.array([ point[0], point[1] ])
			x2 = np.array([ point[2], point[3] ])
			x3 = np.array([ point[4], point[5] ])
			X0 = np.array([ point[6] , point[7], point[8] ]).reshape(3, 1)
			
			re 	= self.single_correspondence_nonlinear_triangulation(self._K, self._C1, self._R1, self._C2, self._R2, self._C3, self._R3, x1, x2, x3, X0)
			re2 = self.single_correspondence_nonlinear_triangulation(self._K, self._C1, self._R1, self._C2, self._R2, self._C3, self._R3, x1, x2, x3, re)
			re3 = self.single_correspondence_nonlinear_triangulation(self._K, self._C1, self._R1, self._C2, self._R2, self._C3, self._R3, x1, x2, x3, re2)

			X[idx] = re3.transpose()

		return X

	def single_correspondence_nonlinear_triangulation(self, K, C1, R1, C2, R2, C3, R3, x1, x2, x3, X0):
		'''
		Inputs:
			K 	- size (3 x 3) camera intrinsics matrix
			C1 	- size (3 x 1) camera 1 pose
			R1 	- size (3 x 3) camera 1 rotation
			C2 	- size (3 x 1) camera 2 pose
			R2 	- size (3 x 3) camera 2 rotation
			C3 	- size (3 x 1) camera 3 pose
			R3 	- size (3 x 3) camera 3 rotation
			x1 	- size (1 x 2) 2D point on frame 1
			x2 	- size (1 x 2) 2D point on frame 2
			x3 	- size (1 x 2) 2D point on frame 3
			X0 	- size (3 x 1) estimated 3D point
		Output:
			X 	- size (3 x 1) corrected 3D point
		'''
		b = np.array([ x1[0], x1[1], x2[0], x2[1], x3[0], x3[1] ]).reshape(6, 1)

		# Reprojected coordinates
		# xi = K * Ri * (X0 - Ci)
		x1_rep = np.matmul((K * R1), (X0 - C1))
		x2_rep = np.matmul((K * R2), (X0 - C2))
		x3_rep = np.matmul((K * R3), (X0 - C3))

		u1, v1, w1 = x1_rep[0], x1_rep[1], x1_rep[2]
		u2, v2, w2 = x2_rep[0], x2_rep[1], x2_rep[2]
		u3, v3, w3 = x3_rep[0], x3_rep[1], x3_rep[2]

		f = np.array([ u1 / w1 , v1 / w1 , u2 / w2 , v2 / w2 , u3 / w3 , v3 / w3 ]).reshape(6, 1)

		J = self.Jacobian_triangulation(K, C1, R1, C2, R2, C3, R3, X0)
		
		#  Explanation: delta_X = (J^T * J).inv * J^T * (b - f(X))
		JT = J.transpose()
		JT_J_inv = inv( np.dot( JT, J) )
		delta_X = multi_dot( [JT_J_inv, JT, (b - f)] )

		return X0 + delta_X


	def Jacobian_triangulation(self, K, C1, R1, C2, R2, C3, R3, X0):
		'''
		Output:
			J - size (6 x 3) The Jacobian for all three points
				[ df1dX
				  df2dX
				  df3dX ]
		'''
		df1dX = self.dfdX(C1, R1, K, X0)
		df2dX = self.dfdX(C2, R2, K, X0)
		df3dX = self.dfdX(C3, R3, K, X0)

		return np.concatenate( (df1dX, df2dX, df3dX), axis=0)

	def dfdX(self, C, R, K, X):
		'''
		Inputs:
			C - size (3 x 1) Camera pose matrix (cx, cy, cz)^T
			R - size (3 x 3) rotation matrix
				r11  r12  r13
				r21  r22  r23
				r31  r32  r33
			K - size (3 x 3) camera intrinsics matrix:
				f  0  px
				0  f  py
				0  0  1
			X - size (3 x 1) estimated 3D points (x, y, z)^T
		Outputs:
			dfdX - size(2 x 3) the partial derivative of f on X

			dfdX = [ (w * dudX - w * dwdX) / w^2
					 (w * dvdX - v * dwdX) / w^2 ]
		'''
		f 	= K[0, 0]
		px 	= K[0, 2]
		py 	= K[1, 2]

		r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
		r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
		r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

		dudX = np.array([f * r11 + px * r31 , f * r12 + px * r32 , f * r13 + px * r33 ])
		dvdX = np.array([f * r21 + py * r31 , f * r22 + py * r32 , f * r23 + py * r33 ])
		dwdX = np.array([r31 , r32 , r33])

		# x = K * R * (X - C)   x has size 3 x 1
		x = multi_dot([K, R, (X - C)])

		u = x[0, 0]
		v = x[1, 0]
		w = x[2, 0]

		dfdX1 = (w * dudX - u * dwdX) / w**2
		dfdX1 = dfdX1.reshape(1, 3)
		dfdX2 = (w * dvdX - v * dwdX) / w**2
		dfdX2 = dfdX2.reshape(1, 3)

		return np.concatenate( (dfdX1, dfdX2), axis=0)