# -*- coding: utf-8 -*-
"""
BIOE 5100 Final Project
"""

import skimage as exSki # https://scikit-image.org/docs/stable/api/skimage.html
import imageio.v3 as exIio # Recommended to use imageIO for io functionality instead of SKImage, since SKImage's IO is now a wrapper for ImageIO.

import numpy as np
from numpy import fft # I don't think it matters whether we use NumPy's implementation of FFT or SciPy's.
import matplotlib.pyplot as plt # For histogram visualization, etc.

def main():
    # read in all tif images into a list
    lImageSet = list(exSki.io.imread_collection('images/*.tif', {'as_gray': True}))
    
    # show a test image just because
    exSki.io.imshow(lImageSet[9], cmap='gray')
    plt.title('image #9')
    exSki.io.show()
    
    # find the max image size in the collection
    lMaxImageRows, lMaxImageCols = 0, 0    
    for iImage in lImageSet:
        lCurrentImageRows, lCurrentImageCols = iImage.shape[0], iImage.shape[1]
        if iImage.shape[0] > lMaxImageRows: lMaxImageRows = iImage.shape[0]
        if iImage.shape[1] > lMaxImageCols: lMaxImageCols = iImage.shape[1]
        
    # now that we have the max sizes, let's upsize everything to that
    for iImageIndex in range(len(lImageSet)):
        lImageSet[iImageIndex] = exSki.transform.resize(lImageSet[iImageIndex], (lMaxImageRows, lMaxImageCols))
    
    # show the test image just because
    exSki.io.imshow(lImageSet[9], cmap='gray')
    plt.title('image #9 resized')
    exSki.io.show()
    
    # ok... now we want to just throw it into a model as it is, w/o pre-processing
        
    # actually I'm just going to feed one image to a kmeans clustering model and see if it can distinguish plaques  
    lTestImage = lImageSet[9].reshape(-1,1)
    from sklearn.cluster import KMeans
    lKMeansModel = KMeans(n_clusters=2, random_state=42)
    lKMeansModel.fit(lTestImage)
    lKMeansPredictedLabels = lKMeansModel.predict(lTestImage).reshape(lImageSet[9].shape)
    #lKMeansClusteredImage = np.zeros(lImageSet[9].shape)
    #lKMeansClusteredImage = lKMeansPredictedLabels + 1
    exSki.io.imshow(lKMeansPredictedLabels, cmap='plasma')
    plt.title('image #9 clustered')
    exSki.io.show()
    
    
    print()







if __name__=="__main__":
    main()
