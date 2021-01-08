#!/usr/bin/env python
#!/usr/bin/python

from orb import *
from fundamental_matrix import *
import cv2
import numpy as np

if __name__ == '__main__':
	orb = ORB()

	test_img_1 = cv2.imread('/home/charly_huang/Downloads/KITTI_datasets/01/image_0/000000.png', cv2.IMREAD_GRAYSCALE)
	kps1, descrs1 = orb.compute(test_img_1)

	orb.draw(test_img_1, kps1)

	test_img_2 = cv2.imread('/home/charly_huang/Downloads/KITTI_datasets/01/image_0/000001.png', cv2.IMREAD_GRAYSCALE)
	kps2, descrs2 = orb.compute(test_img_2)

	matches = orb.match(descrs1, descrs2)
	# sort the matches according to their disntaces
	matches = sorted( matches, key=lambda x: x.distance )

	orb.draw_matched(test_img_1, kps1, descrs1, test_img_2, kps2, descrs2)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Find fundamental matrix
	# extract the matched keypoints
	matched_kps1 = np.float32([kps1[m.idx].pt for m in matches]).reshape(-1, 1, 2)
	matched_kps2 = np.float32([kps2[m.idx].pt for m in matches]).reshape(-1, 1, 2)

	# find homography matrix and do perspective transform
	