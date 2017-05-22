import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import os

"""
This scripts find the axes of an object
"""

"""
First Method: Covariance matrix and eigens
"""
# 1.Matrix of points:
#   We'll be dealing with 2D points so our matrix is 2xm. The 1st row are the x-coordinates,
#   2nd row are the y-coordinates and m indicates the number of points.
os.chdir("/home/mrobot/Documentos/TFG/code/Skin-Lesion-Analysis-Towards-Melanoma-Detection")
img = scipy.misc.imread('image/train_ground/ISIC_0000000_segmentation.png', flatten=True)
y, x = np.nonzero(img)

# 2.Subtract the mean for each point:
#   Calculate the mean of the row of x-coordinates. For each x-point, subtract the mean from it. Do the same for
#   the y-coordinate. Mean subtraction minimises the mean square error of approximating the data and centers the data.

x = x - np.mean(x)
y = y - np.mean(y)
coords = np.vstack([x,y])

# 3.Covariance matrix calculation:
#   Calculate the 2x2 covariance matrix.

cov = np.cov(coords)

# 4.Eigenvector, eigenvalues of covariance matrix:
#   Find the 2 eigen-pairs of our dataset.

evals, evecs = np.linalg.eig(cov)

# 5.Rearrange the eigen-pairs:
#   Sort by decreasing eigenvalues

sort_indices = np.argsort(evals)[::-1]
evec1, evec2 = evecs[:, sort_indices]
x_v1, y_v1 = evec1
x_v2, y_v2 = evec2

# Plot tumor
plt.plot(x, y, 'k.')

# Rotation of the tumor
theta = np.arctan((x_v1)/(y_v1))
rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
transformed_mat = rotation_mat * coords
x_transformed, y_transformed = transformed_mat.A

plt.plot(x_transformed, y_transformed, 'g.')


# Obtener el área que no se solapa con su reflexión
"""
t = [[1, 0]]
d = t * transformed_mat
index_over = np.where(d>=0)[1]
x_over = x_transformed[index_over]
y_over = y_transformed[index_over]

index_sub = np.where(d<0)[1]
x_sub = x_transformed[index_sub]
y_sub = y_transformed[index_sub]

plt.plot(x_over*-1, y_over, 'y.')
plt.plot(x_sub*-1, y_sub, 'b.')
"""
x_transformed_reflection = x_transformed*-1
y_transformed_reflection = y_transformed

x_transformed_reflection = x_transformed_reflection.astype(int)
y_transformed_reflection = y_transformed_reflection.astype(int)

index =  np.array([])

for idx, val in enumerate(x_transformed_reflection):
    print(idx, val)
    if val == x_transformed[idx]:
        index = np.append(index, idx)

plt.plot(x_transformed_reflection, y_transformed_reflection, 'y.')

equal_x = np.not_equal(x_transformed.astype(int), x_transformed_reflection.astype(int)).nonzero()[0]

x_overlapping = x_transformed[equal_x]
y_overlapping = y_transformed[equal_x]

#plt.plot(x_overlapping, y_overlapping, 'r.')


#d = (y_v1)*x0 + (-x_v1)*y0 # determines if a certain point (x0,y0) is over(d>0) or sub(d<0) to the straight line
#print(d)
#plt.plot([x0], [y0], 'wo')

# 6.Plot the principal components
scale = 60



plt.plot([x_v1*-scale*2, x_v1*scale*2],
         [y_v1*-scale*2, y_v1*scale*2], color='red')

plt.plot([x_v2*-scale, x_v2*scale],
         [y_v2*-scale, y_v2*scale], color='blue')

plt.plot([0], [0], 'ro')
plt.plot([0], [100], 'wo')

plt.axis('equal')
plt.gca().invert_yaxis()
plt.show()