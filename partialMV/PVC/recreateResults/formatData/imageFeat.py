from skimage.feature import hog
import scipy.io
import numpy as np

feat = {}
mat = scipy.io.loadmat('orl.mat')
feat['hogs'] = []
feat['fea'] = mat['fea'].astype(float)
feat['gnd'] = mat['gnd'].astype(float)

mat = mat['fea']

for i in mat:
	i = np.array(i)
	img = i.reshape((32,32))
	fd = hog(img, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=False)
	feat['hogs'].append(fd)

feat['hogs'] = np.array(feat['hogs'])

scipy.io.savemat('orlHog.mat', mdict={'feat': feat})