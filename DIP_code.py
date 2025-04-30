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
    exSki.io.imshow(lImageSet[9], cmap='gray')# TODO: Change to imageIO.
    exSki.io.show() # TODO: Change to imageIO.
    
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
    exSki.io.imshow(lImageSet[9], cmap='gray')# TODO: Change to imageIO.
    exSki.io.show() # TODO: Change to imageIO.
    
    # ok... now we want to just throw it into a model as it is, w/o pre-processing
        
      
    
    
    print()







if __name__=="__main__":
    main()
