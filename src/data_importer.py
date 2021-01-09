#!/usr/bin/env python
#!/usr/bin/python

import numpy as np
import pandas as pd
import h5py
import cv2

class Data_Importer:
	'''
	This class is used exclusively to import the PennU dataset 
	that I exported from Assignment 4 of Multiview Geometry open course.
	'''
	def __init__(self, 
				 x1_file = None, 
				 x2_file = None, 
				 x3_file = None, 
				 C_file = None,
				 R_file = None, 
				 K_file = None,
				 img1_file = None,
				 img2_file = None,
				 img3_file = None):
		self._x1_file = x1_file
		self._x2_file = x2_file
		self._x3_file = x3_file
		self._C_file = C_file
		self._R_file = R_file
		self._K_file = K_file
		self._img1_file = img1_file
		self._img2_file = img2_file
		self._img3_file = img3_file

		checklist = [('x1_file', self._x1_file), ('x2_file', self._x2_file), ('x3_file', self._x3_file), \
					 ('C_file', self._C_file), ('R_file', self._R_file), ('K_file', self._K_file), \
					 ('img1_file', self._img1_file), ('img2_file', self._img2_file), ('img3_file', self._img3_file)]
		checks = len(checklist)
		for varName, var in checklist:
			if var is None:
				print('{} has not yet been assigned. Please specify a directory to it.\n\n'.format(varName))
				checks -= 1
		if checks == len(checklist):
			print('All dataset directories are properly set.')


	def set_x1_src(self, x1_file):
		self._x1_file = x1_file

	def set_x2_src(self, x2_file):
		self._x2_file = x2_file

	def set_x3_src(self, x3_file):
		self._x3_file = x3_file

	def set_C_src(self, C_file):
		self._C_file = C_file

	def set_R_src(self, R_file):
		self._R_file = R_file

	def set_K_src(self, K_file):
		self._K_file = K_file

	def set_img1_src(self, img1_file):
		self._img1_file = img1_file

	def set_img2_src(self, img2_file):
		self._img2_file = img2_file

	def set_img3_src(self, img3_file):
		self._img3_file = img3_file

	def render(self):
		# x1, x2, x3
		x1 = self.render_from_csv(self._x1_file)
		x2 = self.render_from_csv(self._x2_file)
		x3 = self.render_from_csv(self._x3_file)

		# Camera center C, Roration R, Intrinsic params K
		C = self.render_matrix_from_csv(self._C_file)
		## The camera pose should appear as a 3 x 1 vector
		C = C.reshape(3, 1)

		R = self.render_matrix_from_csv(self._R_file)
		K = self.render_matrix_from_csv(self._K_file)

		# img1, img2, img3
		img1 = self.render_from_h5(self._img1_file)
		img2 = self.render_from_h5(self._img2_file)
		img3 = self.render_from_h5(self._img3_file)

		return x1, x2, x3, C, R, K, img1, img2, img3

	def render_from_csv(self, file):
		return pd.read_csv(file)

	def render_matrix_from_csv(self, file):
		return np.loadtxt( open(file, 'rb'), delimiter=',')

	def render_from_h5(self, file):
		file = h5py.File(file, 'r')
		data = file.get( file.keys()[0] )
		npDataset = np.array( data.get('value') )

		channels, height, width = npDataset.shape
		# Convert to cv2 matrix
		toImg = np.zeros([height, width, channels])

		for channel in range(channels):
			toImg[:, :, channel] = npDataset[channel, :, :]/255.
		toImg = cv2.rotate( toImg, cv2.ROTATE_90_CLOCKWISE )

		return toImg