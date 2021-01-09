#!/usr/bin/env python
#!/usr/bin/python

from os.path import dirname, join, isdir
import rospy
from data_importer import *
from fundamental_matrix import *
from essential_matrix import *

if __name__=='__main__':
	rospy.init_node('py_sfm', anonymous=True)

	print('Launched py_sfm')

	datasets_dir = join( dirname( dirname( __file__ ) ), 'penn_u_datasets' )
	if not isdir(datasets_dir):
		raise Exception('The provided dataset directory is not valid. Please check again.')
	importer = Data_Importer(x1_file=join(datasets_dir, 'x1.csv'),
							 x2_file=join(datasets_dir, 'x2.csv'),
							 x3_file=join(datasets_dir, 'x3.csv'),
							 C_file=join(datasets_dir, 'C.csv'),
							 R_file=join(datasets_dir, 'R.csv'),
							 K_file=join(datasets_dir, 'K.csv'),
							 img1_file=join(datasets_dir, 'img1.h5'),
							 img2_file=join(datasets_dir, 'img2.h5'),
							 img3_file=join(datasets_dir, 'img3.h5'))

	x1, x2, x3, C, R, K, img1, img2, img3 = importer.render()

	# Estimate fundamental matrix
	FundMat = Fundamental_matrix(x1, x2)
	F = FundMat.estimate()
	print(F)

	# Estimate essential matrix from fundamental matrix ( and thus we have T and R)
	EssMat = Essential_matrix(F, K)
	E = EssMat.estimate()

	print(E)

	# Obtain 3D points using correct camera pose (Linear Triangulation)

	# Find the third camera pose using Linear PnP

	# Calculate reprojection point -> visualization

	# Visualization

	# Nonelinear triangulation

	# Display point cloud and three camera poses

	# Calculate reprojection points -> visualization

	# Display correspondences between the keypoints and reprojection

	# rospy.spin()