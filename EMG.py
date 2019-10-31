'''
INTRODUCTION TO MACHINE LEARNING
NAME: MOHANA KRISHNA VUTUKURU
EXPECTATION MAXIMIZATION
'''
# 1. Run KMeans
# 2. 
from skimage import io
from skimage.viewer import ImageViewer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def EMG(image, k, flag):

    # Turn the image into pixels
    image = image[:,:,:3]/255
    pixels = np.reshape(image[:,:,:3], (image.shape[0]*image.shape[1], 3))    
    colors = KMeans(n_clusters=k).fit(pixels)
    clusters = np.hstack((pixels, np.reshape(colors.labels_, (13400,1))))
    
    data_dict, means, variances = {}, {}, {}
    for i in range(k):
        data_dict[i] = clusters[clusters[:,-1] == i][:,:-1]
        means[i] = np.mean(data_dict[i], axis = 0)
        variances[i] = np.cov(data_dict[i], rowvar=False)
    return variances

img = io.imread('stadium.bmp')
print(EMG(img, 5, 0))