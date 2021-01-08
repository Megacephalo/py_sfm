#!/usr/bin/env python
#!/usr/bin/python

import cv2

class Feature_Matcher_base:
	def __init__(self):
		'''
		Virtual function.
		Instantiate the detector in the constructor
		'''
		self.matches = None

	def compute(self, image):
		'''
		Virtual function
		'''
		keypoints, descriptors = None, None
		return keypoints, descriptors

	def match(self, descriptors_1, descriptors_2):
		# Use Brute-force matcher
		matcher = cv2.BFMatcher()
		self.matches = matcher.match(descriptors_1, descriptors_2)

		return self.matches

	def draw_matched(self, img_1, keypoints_1, descriptors_1, img_2, keypoints_2, descriptors_2):
		if self.matches is None:
			print('There are no matches to draw from.')
			return

		matching_imgs = cv2.drawMatches(img_1, keypoints_1, img_2, keypoints_2,self.matches[:50], None)
		# matching_imgs = cv2.resize(matching_imgs, (1000, 650))
		cv2.imshow('Matches', matching_imgs)
		# cv2.waitKey(0)

	def draw(self, image, keypoints):
		'''
		Draw only keypoint locations, not their sizes nor orientations
		'''
		green = (0, 255, 0)
		render_img = cv2.drawKeypoints(image, keypoints, green, flags=0)
		cv2.imshow('image_with_keypoints', render_img)
		# cv2.waitKey(0)