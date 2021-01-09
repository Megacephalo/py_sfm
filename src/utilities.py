#!/usr/bin/env python
#!/usr/bin/python

import numpy as np

def vec2skew(vector):
	'''
	Input:
		vector - size(3 x 1) The vectorized pose in the form of np.array([x, y, 1])
	Output:
		X - size(3 x 3) The skew symmetric matrix equivalent of the vector
	'''
	if vector is None:
		raise Exception('The given vector is empty')
	if vector.shape[0] != 3:
		raise Exception('The given vector does not comply with the 3D dimension.')

	u12 = -1 * vector[2]
	u13 = vector[1]
	u21 = vector[2]
	u23 = -1 * vector[0]
	u31 = -1 * vector[1]
	u32 = vector[0]
	return np.array([[0, u12, u13],
					[u21, 0, u23],
					[u31, u32, 0]])