from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

def EMG(image, k, flag):

    if flag == 0:
        image = image[:,:,:3]/255
        pixels = np.reshape(image[:,:,:3], (image.shape[0]*image.shape[1], 3))
        colours = KMeans(n_clusters = k).fit(pixels)
        clusters = np.hstack((pixels, np.reshape(colours.labels_, (13400, 1))))
        means = colours.cluster_centers_
        
        # Calculate the initial variances
        variances = [np.cov(clusters[clusters[:,-1]==i][:,:-1], rowvar = False) for i in range(k)]
        pi = [len(clusters[clusters[:,-1]==i])/len(pixels) for i in range(k)]
        expectation = None
        
        for iterations in range(4):
            for pixel in range(len(pixels)):

                # Expectation step
                likelihood = [multivariate_normal.pdf(pixels[pixel], means[gaussian], variances[gaussian]) for gaussian in range(k)]
                expectation = np.multiply(pi, likelihood)/np.sum(np.multiply(pi, likelihood))
                clusters[:,-1][pixel] = np.argmax(expectation)

                # Maximization step
                pi = [len(clusters[clusters[:,-1]==i])/len(pixels) for i in range(k)]
                means = [np.mean(clusters[clusters[:,-1]==i][:,:3], axis = 0) for i in range(k)]
                variances = [np.cov(clusters[clusters[:,-1] == i][:,:-1], rowvar = False) for i in range(k)]

            print(iterations,':',means)

        compressed_image = np.array(means)[clusters[:,-1].astype(int)]
        compressed_image = np.reshape(compressed_image, (image.shape[0],image.shape[1], 3))
        plt.imshow(compressed_image)
        plt.show()
        return means
    
    if flag == 1:
        pass

img = io.imread('stadium.bmp')
print(EMG(img, 4, 0))
