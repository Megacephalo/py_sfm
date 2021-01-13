#!/usr/bin/env python
#!/usr/bin/python

from os.path import dirname, join, isdir
import rospy
from data_importer import *
from fundamental_matrix import *
from essential_matrix import *
from linear_triangulation import *
from linearPnP import *
from nonlinear_triangulation import *
from Display3D import *

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

	# Estimate essential matrix from fundamental matrix ( and thus we have T and R)
	EssMat = Essential_matrix(F, K)
	E = EssMat.estimate()

	# Obtain 3D points using correct camera pose (Linear Triangulation)
	## NOTE: The normal procedure is to obtain C1, R1, C2, R2 from the essential matrix,
	## But the assignment is saving the effort by directly providing these variables from the datasets.
	## We can also understand it this way: we are estimating from the first to second frame of 
	## the entire video sequence, hence the initial state would be C = zeros(3, 1) and R = identity(3).
	linTriag = Linear_triangulation(K, np.zeros((3, 1)), np.identity(3), C, R, x1, x2)
	threeD_pts = linTriag.estimate()

	# Find the third camera pose using Linear PnP
	linearPnP = LinearPnP(threeD_pts, x3, K)
	C3, R3 = linearPnP.estimate()

	# Nonelinear triangulation
	nonLinearTriag = NonLinear_Triangulation(K, 
											 np.zeros((3, 1)), np.identity(3), 
											 C, R, 
											 C3, R3, 
											 x1, x2, x3, threeD_pts)
	threeD_pts = nonLinearTriag.correct()

	# Display point cloud and three camera poses
	# Cset: C1, C2, C3
	# Rset: R1, R2, R3
	imgTuple = (join(datasets_dir, 'image1.png'), join(datasets_dir, 'image2.png'), join(datasets_dir, 'image3.png'))
	display = Display3D((np.zeros((3, 1)), C, C3), 
						(np.identity(3), R, R3), 
						threeD_pts, 
						imgTuple)
	display.show()