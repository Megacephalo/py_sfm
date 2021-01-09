#!/usr/bin/env python
#!/usr/bin/python

import numpy as np
from numpy.linalg import svd

class Essential_matrix:
	'''
	Use the camera calibration matrix to estimate the Essential matrix
	Inputs:
		K - size (3 x 3) camera calibration (intrinsic) matrix
		F - size (3 x 3) fundamental matrix
	Outputs:
		E - size (3 x 3) essential matrix with singular alues (1, 1, 0)
	'''
	def __init__(self, FundamentalMatrix, CameraMatrix):
		self._F = FundamentalMatrix
		self._K = CameraMatrix

		if not self.isValidMatrix(self._F):
			raise Exception('The given fundamental matrix does not have dimensions 3 x 3')
		if not self.isValidMatrix(self._K):
			raise Exception('The given essential matrix does not have dimensions 3 x 3')

	def estimate(self):
		if not self.isValidMatrix(self._F):
			raise Exception('The given fundamental matrix does not have dimensions 3 x 3')
		if not self.isValidMatrix(self._K):
			raise Exception('The given essential matrix does not have dimensions 3 x 3')

		E1 = np.transpose(self._K) * self._F * self._K

		U, __, V = svd(E1)

		print('U: ', U)
		print('V: ', V)

		E = U * np.array([(1, 0, 0), (0, 1, 0), (0, 0, 0)]) * np.transpose(V)

		return E

	def isValidMatrix(self, matrix):
		'''
		Verify that the iinput matrix has dimensions 3 x 3
		'''
		rows, cols = matrix.shape
		if rows != 3 or cols != 3:
			return False

		return True