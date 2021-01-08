#!/usr/bin/env python
#!/usr/bin/python

from feature_matcher_base import *
import cv2

class ORB(Feature_Matcher_base):
	def __init__(self):
		Feature_Matcher_base.__init__(self)
		self._orb = cv2.ORB_create()

	def compute(self, image):
		'''
		overriding method
		'''
		keypoints, descriptors = self._orb.detectAndCompute(image, None)

		return keypoints, descriptors

	


