# -*- coding: utf-8 -*-
"""
BIOE 5100 Final Project
"""

import skimage as exSki # https://scikit-image.org/docs/stable/api/skimage.html
import imageio.v3 as exIio # Recommended to use imageIO for io functionality instead of SKImage, since SKImage's IO is now a wrapper for ImageIO.

import numpy as np
from sklearn.cluster import KMeans
from numpy import fft # I don't think it matters whether we use NumPy's implementation of FFT or SciPy's.
import matplotlib.pyplot as plt # For histogram visualization, etc.

def main():
    # read in all tif images into a list
    lImageSet = list(exSki.io.imread_collection('images/*.tif', {'as_gray': True}))
    
    # show a test image just because
    exSki.io.imshow(lImageSet[9], cmap='gray')
    plt.title('image #9')
    exSki.io.show()
    
    lImageSet = fUpsizeImagesToLargestInSet(lImageSet)
    
    # ok... now we want to just throw it into a model as it is, w/o pre-processing, see what happens
        
    # out of curiosity, show image 9 and its histogram
    h = exSki.exposure.histogram(lImageSet[9], normalize=True)
    figure, axes = plt.subplots(1,2, layout='tight')
    figure.suptitle('Image #9')
    axes[0].imshow(lImageSet[9], cmap='gray')
    axes[1].plot(h[1], h[0])
    plt.show()
    
    # try histogram equalization for higher contrast
    # SOOO doing that makes the noise louder (and therefore kmeans doesn't work as well)... so no.
    lEqualizedImage = exSki.exposure.equalize_hist(lImageSet[9])
    h = exSki.exposure.histogram(lEqualizedImage, normalize=True)
    figure, axes = plt.subplots(1,2, layout='tight')
    figure.suptitle('Equalized Image #9')
    axes[0].imshow(lEqualizedImage, cmap='gray')
    axes[1].plot(h[1], h[0])
    plt.show()
    
    # let's try a median filter to remove noise
    lMedImage = exSki.filters.median(lImageSet[9])
    h = exSki.exposure.histogram(lMedImage, normalize=True)
    figure, axes = plt.subplots(1,2, layout='tight')
    figure.suptitle('Median Filtered Image #9')
    axes[0].imshow(lMedImage, cmap='gray')
    axes[1].plot(h[1], h[0])
    plt.show()
    
    
    # I'm just going to feed one image to a kmeans clustering model and see if it can distinguish plaques  
    lTestImage = lMedImage
    lKMeansClusteredImage = fDoKMeansClusteringOnImage(lTestImage, 2)
    exSki.io.imshow(lKMeansClusteredImage, cmap='plasma')
    plt.title('image #9 clustered')
    exSki.io.show()
    
    
    print()






def fUpsizeImagesToLargestInSet(aImageSet): 
    lMaxImageRows, lMaxImageCols = 0, 0    
    for iImage in aImageSet: # find the largest image size in the set
        lCurrentImageRows, lCurrentImageCols = iImage.shape[0], iImage.shape[1]
        if iImage.shape[0] > lMaxImageRows: lMaxImageRows = iImage.shape[0]
        if iImage.shape[1] > lMaxImageCols: lMaxImageCols = iImage.shape[1]
    for iImageIndex in range(len(aImageSet)): # now that we have the max sizes, upsize everything to it
        aImageSet[iImageIndex] = exSki.transform.resize(aImageSet[iImageIndex], (lMaxImageRows, lMaxImageCols))
    return aImageSet

def fDoKMeansClusteringOnImage(aImage, aK):
    lImageShape = aImage.shape
    lImage = aImage.reshape(-1,1)
    lKMeansModel = KMeans(n_clusters=aK, random_state=42)
    lKMeansModel.fit(lImage)
    return lKMeansModel.predict(lImage).reshape(lImageShape)
    
if __name__=="__main__":
    main()
