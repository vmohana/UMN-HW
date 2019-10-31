'''
NAME: MOHANA KRISHNA VUTUKURU
DRIVER SCRIPT - HOMEWORK 3
'''

import numpy as np
from sklearn.cluster import KMeans
from skimage import io
import matplotlib.pyplot as plt

# Read the image
img = io.imread('stadium.bmp')
img = img[:,:,:3]/255

# Reshape image
original_shape = img.shape
pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3))

# Fit the KMeans
colors = KMeans(n_clusters=7)
colors = colors.fit(pixels)
compressed_image = colors.cluster_centers_[colors.labels_]
compressed_image = np.reshape(compressed_image, (original_shape))
plt.imshow(compressed_image)
plt.show()
