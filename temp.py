
# Library Imports
import skimage as ski # https://scikit-image.org/docs/stable/api/skimage.html
import imageio.v3 as iio # Recommended to use imageIO for io functionality instead of SKImage, since SKImage's IO is now a wrapper for ImageIO.

import numpy as np
from numpy import fft # I don't think it matters whether we use NumPy's implementation of FFT or SciPy's.
import matplotlib.pyplot as plt # For histogram visualization, etc.


# Importing images from dataset.
directory = 'C:/Users/Larry/Documents/GitHub/DIP_finproj_25/images/*.tif'
imageSet = ski.io.imread_collection(directory) #TODO: Change to imageIO.


# Image re-scaling/cropping?
# Would it make more sense to just do a generic windowing of all images without taking features into account 
# (probably shit idea but did not look through all images)
# or do Canny edge detection and then crop to a standard size based on that?


# Outputting image(s).
testImage = ski.img_as_float64(imageSet[9]) # Using a color image from the dataset. 
# testImage = ski.color.rgb2gray(imageSet[9]) # TODO: Figure out why rgb2gray doesn't work. Probably doesn't matter if using imageio instead of skimage.io.
# testImage = ski.color.rgb2gray(ski.data.astronaut()) # Using an image from the skimage dataset.
ski.io.imshow(testImage)# TODO: Change to imageIO.
ski.io.show() # TODO: Change to imageIO.


# Outputting frequency spectrum of image(s) to determine if frequency filtering is sensible. 
testFFTOutput = np.abs(fft.fftshift(fft.fft2(testImage))) # fft2() is 2D FFT, fftshift() used for same rationale as in MATLAB.
ski.io.imshow(np.log(testFFTOutput), cmap = 'Blues') # Log needs to be used here or else fft is uninterpretable.
ski.io.show()
# From the above, I don't think we should do frequency filtering, but instead go straight to smoothing, edge detection, and then regression/whatever.

# Applying a median filter.
medianFilteredImg = ski.filters.median(testImage, mode = 'nearest')
figure, axes = plt.subplots(nrows = 1, ncols = 2)
axes[0].imshow(testImage)
axes[0].set_title('Original')
axes[1].imshow(medianFilteredImg)
axes[1].set_title('Median Filtered')
plt.show()

# Histogram visualization of a subset(?) of images.
histogramData, binCenters = ski.exposure.histogram(testImage, nbins = 256) # I assume we'll want to do everything in grayscale for our images? Let me know if that's a bad assumption.
plt.figure() # TODO: Play around with the plot sizes based on what looks good.
plt.plot(binCenters, histogramData)
plt.show()

# Histogram equalization of images.
testImageEqualized = ski.exposure.equalize_hist(testImage, nbins = 256)
figure, axes = plt.subplots(nrows = 1, ncols = 2)
axes[0].imshow(testImage)
axes[0].set_title('Original')
axes[1].imshow(testImageEqualized)
axes[1].set_title('Histogram Equalized')
plt.show()

medianImgEqualized = ski.exposure.equalize_hist(medianFilteredImg, nbins = 256)
figure, axes = plt.subplots(nrows = 1, ncols = 2)
axes[0].imshow(testImage)
axes[0].set_title('Original')
axes[1].imshow(medianImgEqualized)
axes[1].set_title('Median-Filtered Image Equalized')
plt.show()

# Trying gamma correction on the histogram-equalized test image.
# Apologies for how ugly the plots look at present.
testImageGammaCor = ski.exposure.adjust_gamma(testImage, gain = 4)
testImageGammaCorEq = ski.exposure.adjust_gamma(testImageEqualized, gain = 0.05)
figure, axes = plt.subplots(nrows = 1, ncols = 4)
axes[0].imshow(testImage)
axes[0].set_title('Original')
axes[1].imshow(testImageEqualized)
axes[1].set_title('Hist Equal')
axes[2].imshow(testImageGammaCor)
axes[2].set_title('Org.+Gamma')
axes[3].imshow(testImageGammaCorEq)
axes[3].set_title('HistEqual+Gamma')
plt.show()

# Canny edge-detection algorithm. Trying on the gamma-corrected, histogram-equalized version of the image.
# TODO: Figure out a good way to clean the images such that we get more sensible edge-detection.
# Or disregard edge-detection for another approach.
imageEdges = ski.feature.canny(testImageGammaCor, sigma = 0.5, low_threshold = 0.1, high_threshold = 0.5)
figure, axes = plt.subplots(nrows = 1, ncols = 2)
axes[0].imshow(testImage)
axes[0].set_title('Original')
axes[1].imshow(imageEdges)
axes[1].set_title('Edges Detected by Canny')
plt.show()

# TBD Classification method
