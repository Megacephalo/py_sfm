#!/usr/bin/env python
#!/usr/bin/python

from video_reader import *

if __name__=='__main__':
	src = '/home/charly_huang/Downloads/KITTI_datasets/01/image_0/'
	reader = Video_reader(src)
	print(reader)

	tempList = reader.render()
	print('There are {} frames stored in the list.'.format( len(tempList) ))

	print('Playing the frames...')
	reader.play_sequence(tempList)
	print('Done playing.')

	