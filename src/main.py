#!/usr/bin/env python
#!/usr/bin/python

import rospy

if __name__=='__main__':
	rospy.init_node('py_sfm', anonymous=True)

	print('Launched py_sfm')

	rospy.spin()