# -*- coding: utf-8 -*-
"""
BIOE 5100 Final Project - Spring 2025
Skylar Suarez & Larry Baker 

Variable naming convention: lLocalVariable, fUserDefinedFunction, exExternalLibrary, aArgument, iIterator
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
    
    # exSki.io.imshow_collection(lImageSet, plugin='matplotlib', cmap='gray')
    # exSki.io.show()
    
    # show a test image just because
    plt.imshow(lImageSet[9], cmap='gray')
    plt.title('image #9')
    plt.show()
    
    lImageSet = fUpsizeImagesToLargestInSet(lImageSet)
    
    # ok... now we want to just throw it into a model as it is, w/o pre-processing, see what happens
        
    # out of curiosity, show image 9 and its histogram
    #fDisplayImageAndItsHistogram(lImageSet[9], 'Image #9')
    
    # try histogram equalization for higher contrast
    # SOOO doing that makes the noise louder (and therefore kmeans doesn't work as well)... so no.
    lEqualizedImage = exSki.exposure.equalize_hist(lImageSet[16])
    fDisplayImageAndItsHistogram(lEqualizedImage, 'Equalized Image #9')
        
    # let's try a median filter to remove noise
    lMedImage = exSki.filters.median(lImageSet[16])
    fDisplayImageAndItsHistogram(lMedImage, 'Median Filtered Image #9')
    
    # let's try a gaussian filter to remove noise
    lGaussImage = exSki.filters.gaussian(lImageSet[16], 0.5)
    fDisplayImageAndItsHistogram(lGaussImage, 'Gaussian Filtered Image, sigma = 0.5')

        
    
    # I'm just going to feed one image to a kmeans clustering model and see if it can distinguish plaques  
    lTestImage = lImageSet[16]
    lKMeansClusteredImage = fDoKMeansClusteringOnImage(lTestImage, 2)
    plt.imshow(lKMeansClusteredImage, cmap='plasma')
    plt.title('image #9 clustered')
    plt.show()
    
    lTestImage = lGaussImage
    lKMeansClusteredImage = fDoKMeansClusteringOnImage(lTestImage, 2)
    fig, axs = plt.subplots(1,1)
    axs.imshow(lImageSet[16], cmap='gray')
    axs.imshow(lKMeansClusteredImage, cmap='plasma', alpha=0.05)
    plt.title('gaussian image #9 clustered')
    plt.show()
    
    
    # FIRST, find image w/o ab (5 works)
    
    # do kmeans on an image w/o ab, see how different the clustered images are, maybe that can be used for classification?
    lTestNoABGaussImage = exSki.filters.gaussian(lImageSet[5], 0.5)
    lNoABKMeansClusteredImage = fDoKMeansClusteringOnImage(lTestNoABGaussImage, 2)
       
    fig, axes = plt.subplots(2,1, layout='tight')
    axes[0].imshow(lKMeansClusteredImage, cmap='plasma')
    axes[1].imshow(lNoABKMeansClusteredImage, cmap='plasma')
    plt.title('comparing ab to no ab clustered images, both gauss 0.5')
    plt.show()
    
    # do GLCM on image 9, see what the resulting matrix looks like compared to an image w/o ab (find one)
    lGLCM = exSki.feature.graycomatrix(lImageSet[16], distances=[1], angles=[0], levels=np.max(lImageSet[9])+1)
    lGLCM = lGLCM[:, :, 0, 0]
    plt.imshow(lGLCM, cmap='gray')
    plt.show()
    
    print()






def fUpsizeImagesToLargestInSet(aImageSet): 
    lMaxImageRows, lMaxImageCols = 0, 0    
    for iImage in aImageSet: # find the largest image size in the set
        lCurrentImageRows, lCurrentImageCols = iImage.shape[0], iImage.shape[1]
        if iImage.shape[0] > lMaxImageRows: lMaxImageRows = iImage.shape[0]
        if iImage.shape[1] > lMaxImageCols: lMaxImageCols = iImage.shape[1]
    for iImageIndex in range(len(aImageSet)): # now that we have the max sizes, upsize everything to it
        aImageSet[iImageIndex] = exSki.transform.resize(aImageSet[iImageIndex], (lMaxImageRows, lMaxImageCols), preserve_range=True).astype(int)
    return aImageSet

def fDisplayImageAndItsHistogram(aImage, aTitle):
    lHistInfo = exSki.exposure.histogram(aImage, normalize=True)
    lFigure, lAxes = plt.subplots(1,2, layout='tight')
    lFigure.suptitle(aTitle)
    lAxes[0].imshow(aImage, cmap='gray')
    lAxes[1].plot(lHistInfo[1], lHistInfo[0])
    plt.show()

def fDoKMeansClusteringOnImage(aImage, aK):
    lImageShape = aImage.shape
    lImage = aImage.reshape(-1,1)
    lKMeansModel = KMeans(n_clusters=aK, random_state=42)
    lKMeansModel.fit(lImage)
    return lKMeansModel.predict(lImage).reshape(lImageShape)
    
if __name__=="__main__":
    main()
