
# I'm keeping my playing around to this file for now to not crowd the main file too much at first.
# Library Imports
import skimage as ski # https://scikit-image.org/docs/stable/api/skimage.html
import imageio.v3 as iio # Recommended to use imageIO for io functionality instead of SKImage, since SKImage's IO is now a wrapper for ImageIO. I will ignore this lol.

import numpy as np
from numpy import fft # I don't think it matters whether we use NumPy's implementation of FFT or SciPy's.
import matplotlib.pyplot as plt # For histogram visualization, etc.
from matplotlib.widgets import Slider
from sklearn.cluster import KMeans

# Function definitions to make my code more legible.
def medianFiltering(aImage): # Plots and returns a median-filtered image.
    medianFilteredImg = ski.filters.median(aImage, mode = 'nearest')
    figure, axes = plt.subplots(nrows = 1, ncols = 2, layout = 'tight')
    axes[0].imshow(aImage, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(medianFilteredImg, cmap='gray')
    axes[1].set_title('Median Filtered Image')
    plt.show()
    return medianFilteredImg

def histEqualization(aImage):
    imageEqualized = ski.exposure.equalize_hist(aImage, nbins = 256)
    figure, axes = plt.subplots(nrows = 1, ncols = 2, layout = 'tight')
    axes[0].imshow(aImage, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(imageEqualized, cmap='gray')
    axes[1].set_title('Histogram Equalized')
    plt.show()
    return imageEqualized

def fDoKMeansClusteringOnImage(aImage, aK):
    lImageShape = aImage.shape
    lImage = aImage.reshape(-1,1)
    lKMeansModel = KMeans(n_clusters=aK, random_state=42)
    lKMeansModel.fit(lImage)
    return lKMeansModel.predict(lImage).reshape(lImageShape)

def multiOtsuThreshold(aImage, numClasses): # See: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html#sphx-glr-auto-examples-segmentation-plot-multiotsu-py
    thresholds = ski.filters.threshold_multiotsu(aImage, classes = numClasses)
    regions = np.digitize(aImage, bins = thresholds)
    figure, axes = plt.subplots(nrows = 1, ncols = 2, layout = 'tight')
    axes[0].imshow(aImage, cmap = 'gray')
    axes[0].set_title('Original')
    axes[1].imshow(regions, cmap = 'gray')
    axes[1].set_title('Multi-Otsu Thresholding')
    plt.show()
    return

def histeresisThreshold(aImage, lowThreshold, highThreshold): # See: https://scikit-image.org/docs/stable/auto_examples/filters/plot_hysteresis.html
    edges = ski.filters.sobel(aImage)
    histeresisImage = ski.filters.apply_hysteresis_threshold(edges, lowThreshold, highThreshold)    
    figure, axes = plt.subplots(nrows = 1, ncols = 2, layout = 'tight')
    axes[0].imshow(aImage, cmap = 'gray')
    axes[0].set_title('Original')
    axes[1].imshow(histeresisImage, cmap = 'gray')
    axes[1].set_title('Hysteresis Thresholding')
    plt.show()
    return


# Importing images from dataset.
directory = 'C:/Users/Larry/Documents/GitHub/DIP_finproj_25/images/*.tif'
imageSet = ski.io.imread_collection(directory)

# Outputting image(s).
testImage = ski.img_as_float64(imageSet[9]) # Using a color image from the dataset. 
# testImage = ski.color.rgb2gray(imageSet[9]) # TODO: Figure out why rgb2gray doesn't work. Probably doesn't matter if using imageio instead of skimage.io.
ski.io.imshow(testImage, cmap='gray')
ski.io.show() 

# # # Outputting frequency spectrum of image(s) to determine if frequency filtering is sensible. 
# # testFFTOutput = np.abs(fft.fftshift(fft.fft2(testImage))) # fft2() is 2D FFT, fftshift() used for same rationale as in MATLAB.
# # ski.io.imshow(np.log(testFFTOutput), cmap = 'Blues') # Log needs to be used here or else fft is uninterpretable.
# # ski.io.show()
# # # From the above, I don't think we should do frequency filtering, but instead go straight to smoothing, edge detection, and then regression/whatever.

# Applying a median filter.
medianFilteredImg = medianFiltering(testImage)

# Histogram equalization of images.
histEqualizedImg = histEqualization(testImage)

# Attempting Multi-Otsu thresholding.
multiOtsuImg = multiOtsuThreshold(testImage, numClasses = 3)

# Attempting regular old hysteresis thresholding just to see what happens.
hysteresisThresholdImg = histeresisThreshold(medianFilteredImg, 100, 150)


# # Trying gamma correction on the histogram-equalized test image (doesn't work yet).
# def calculateGamma(aImage, newGain):
#     return ski.exposure.adjust_gamma(aImage, newGain)

# initialGain = 1

# figure, axes = plt.subplots(nrows = 2, ncols = 2)
# plt.subplots_adjust(left = 0.15, bottom = 0.25)

# axes[0, 0].imshow(testImage, cmap='gray')
# axes[0, 0].set_title('Original')
# axes[0, 1].imshow(histEqualizedImg, cmap='gray')
# axes[0, 1].set_title('Hist Equal')
# axes[1, 0].imshow(calculateGamma(testImage, initialGain), cmap='gray')
# axes[1, 0].set_title('Org.+Gamma')

# x_axis = plt.axes([0.25, 0.1, 0.65, 0.03])

# gainSlider = Slider(
#     ax = x_axis, 
#     label = 'Gamma Gain, Original', 
#     valmin = 0, 
#     valmax = 10, 
#     valinit = initialGain)

# def updateGammaPlot(val):
#     axes[1.0].imshow(calculateGamma(testImage, val))
#     figure.canvas.draw_idle()

# gainSlider.on_changed(updateGammaPlot)
# plt.show()


# # Canny edge-detection algorithm. Trying on the gamma-corrected, histogram-equalized version of the image.
# # TODO: Figure out a good way to clean the images such that we get more sensible edge-detection.
# # Or disregard edge-detection for another approach.
# imageEdges = ski.feature.canny( [], sigma = 1, low_threshold = 0.1, high_threshold = 0.8)
# figure, axes = plt.subplots(nrows = 1, ncols = 2)
# axes[0].imshow(testImage)
# axes[0].set_title('Original')
# axes[1].imshow(imageEdges)
# axes[1].set_title('Edges Detected by Canny')
# plt.show()


    
