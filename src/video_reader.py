#!/usr/bin/env python
#!/usr/bin/python

'''
video_reader

This class simplifies the reading process 
of video clips and check whether the clip is valid
before rendering it to internal frame-wise 
format.
'''
from os.path import dirname, join, exists
from os import listdir
import cv2


class Video_reader:
	def __init__(self, src_dir = None):
		self.src_dir = join(src_dir, '')
		if not exists(self.src_dir):
			raise Exception('The input directory {} does not exist. Please make sure there is no typo.')

	def set_source(self, input_dir):
		if not exists(input_dir):
			raise Exception('Video_reader: the directory does not exist. Check again!')

		self.src_dir = join(src_dir, '')

	def render(self):
		if self.src_dir is None or not exists(self.src_dir):
			raise Exception('Video_render: Make sure that directory is valid or it can be found')

		frame_list = []
		for img in sorted( listdir(self.src_dir) ):
			full_img_dir = join(self.src_dir, img)

			frame = cv2.imread(full_img_dir)
			if frame is None:
				raise Exception('Image {} is empty. Please check whether the input directory is correct.'.format(full_img_dir))
		
			frame_list.append(frame)

		return frame_list

	def play_sequence(self, frame_list):
		if frame_list is None:
			raise Exception('The input video list is empty or corrupted. Please make sure that the list contains the list of frames.')

		test_sample = frame_list[0]
		if test_sample is None:
			raise Exception('The input list does not contain proper image format files. Please check again.')

		for img in frame_list:
			cv2.imshow('animation', img)
			cv2.waitKey(50)
		cv2.destroyAllWindows()


	def __str__(self):
		return 'The images are from {}'.format(self.src_dir)