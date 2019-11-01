from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

def EMG(image, k, flag):

    # Turn the image into pixels
    if flag == 0:
	    image = image[:,:,:3]/255
	    pixels = np.reshape(image[:,:,:3], (image.shape[0]*image.shape[1], 3))    
	    colors = KMeans(n_clusters=k).fit(pixels)
	    clusters = np.hstack((pixels, np.reshape(colors.labels_, (13400,1))))
	    initial_means = colors.cluster_centers_

	    # Store the initial variances and pi values indexed by gaussian.
	    initial_variances, initial_pi = [], []
	    for i in range(k):
	        initial_variances.append(np.cov(clusters[clusters[:,-1]==i][:,:-1], rowvar = False))
	        initial_pi.append(len(clusters[clusters[:,-1]==i])/len(pixels))
	    
	    for pixel in pixels:
	    	for gaussian in range(k):
	    		likelihood = multivariate_normal.pdf(pixel, initial_means[gaussian], initial_variances[gaussian])
		
		'''
        for pixel in pixels:
			likelihood = [multivariate_normal.pdf(pixel, initial_means[gaussian], initial_variances[gaussian]) for gaussian in range(k)]
	    '''
	    return likelihood

    else:
    	pass
img = io.imread('stadium.bmp')
print(EMG(img, 3, 0))
