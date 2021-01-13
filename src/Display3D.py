#!/usr/bin/env pythom
#!/usr/bin/python

import numpy as np
import struct
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

import tf_conversions
import tf2_ros
import geometry_msgs.msg

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from multiprocessing import Process

from utilities import *

class Display3D:
	'''
	Inputs:
		Cset - tuple of all camera pose matrices. 
		Rset - tuple of all rotation matrices
	'''
	def __init__(self, Cset, Rset, X, imageTuple):
		if len(Cset) != len(Rset):
			raise Exception('The numbers of C and R matrices are unmatched. Check the input, please.')

		if X is None:
			raise Exception('There is no given 3D point to analyze. Aborting.')

		self._Cset = Cset
		self._Rset = Rset
		self._X = X

		self._imageTuple = imageTuple

		# TODO: Make them into rosparams
		self._world_frame = 'map'

		self._tf2_br = tf2_ros.TransformBroadcaster()

	def show(self):
		
		while not rospy.is_shutdown():
			self.show_cameras(self._Cset, self._Rset)
			self.show_3D_points()
			self.show_imgs()
			rospy.sleep(.01)

	def show_cameras(self, Cset, Rset):
		'''
		Broadcast all three camera coordinates in terms of TF frames
		'''
		for idx, (camPose, camRot) in enumerate( zip(Cset, Rset) ):
			tf_br = tf2_ros.TransformBroadcaster()
			cam_tf = geometry_msgs.msg.TransformStamped()
			cam_tf.header.stamp = rospy.Time.now()
			cam_tf.header.frame_id = self._world_frame
			cam_tf.child_frame_id = 'camera_' + str(idx + 1)
			cam_tf.transform.translation.x = camPose[0]
			cam_tf.transform.translation.y = camPose[1]
			cam_tf.transform.translation.z = camPose[2]
			cam_tf.transform.rotation = rotationMatrix_2_quaternion(camRot)
			
			tf_br.sendTransform(cam_tf)

	def show_3D_points(self, frame_id = None):
		r, g, b, a = 0, 0, 0, 255
		rgba = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]

		points = []
		for threeDpoint in self._X:
			x = threeDpoint[0]
			y = threeDpoint[1]
			z = threeDpoint[2]
			point = [x, y, z, rgba]
			points.append(point)

		fields = [PointField('x', 0, PointField.FLOAT32, 1),
		          PointField('y', 4, PointField.FLOAT32, 1),
		          PointField('z', 8, PointField.FLOAT32, 1),
		          PointField('rgba', 12, PointField.UINT32, 1),
		         ]


		header = Header()
		if frame_id is None:
			header.frame_id = self._world_frame
		else:
			header.frame_id = frame_id
		pc2 = point_cloud2.create_cloud(header, fields, points)
		pc2.header.stamp = rospy.Time.now()

		pc_pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2)
		pc_pub.publish(pc2)

	def show_imgs(self):
		bridge = CvBridge()
		for idx, img in enumerate(self._imageTuple):
			cvImg = cv2.imread(img)
			image_msg = bridge.cv2_to_imgmsg(cvImg, 'passthrough')
			topic_name = 'image_' + str(idx + 1) + '_topic'
			img_pub = rospy.Publisher(topic_name, Image, queue_size=2)
			img_pub.publish(image_msg)
