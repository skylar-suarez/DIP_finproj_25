# -*- coding: utf-8 -*-
"""
BIOE 5100 Final Project - Spring 2025
Skylar Suarez & Larry Baker 

Variable naming convention: lLocalVariable, fUserDefinedFunction, exExternalLibrary, aArgument, iIterator
"""

import skimage as exSki 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt # For histogram visualization, etc.

def main():
    # read in all tif images into a list
    lImageSet = list(exSki.io.imread_collection('images/*.tif', {'as_gray': True}))
    
    # show the entire collection for presentation purposes
    # exSki.io.imshow_collection(lImageSet, plugin='matplotlib', cmap='gray')
    # exSki.io.show()
        
    lImageSet, lM, lN = fUpsizeImagesToLargestInSet(lImageSet)
    
    # make the vector of whether the images have AB or not (1 = AB, 0 = no AB)
    lImageSetClassifications = np.ones(len(lImageSet), dtype=int)
    lImageSetClassifications[[5,6,13,14,21,22,28,29,33,34,36,37,38,39,41,42,44,45,46,47,48]] = 0
    
    # display which images are AB and which are not
    plt.figure(figsize=(15, 1.5))
    plt.plot(range(len(lImageSet)), lImageSetClassifications, 'o')
    plt.xticks(range(len(lImageSet)))
    plt.yticks([0,1])
    plt.title('1 = AB, 0 = no AB')
    plt.show()
    
    #%% TESTING STUFF ON A SAMPLE IMAGE, for presentation purposes
    lTestImage = lImageSet[9]
    
    # # try histogram equalization for higher contrast
    # # increasing contrast amplifies the speckling and makes kmeans worse
    # lEqualizedImage = exSki.exposure.equalize_hist(lTestImage)
    # fDisplayImageAndItsHistogram(lEqualizedImage, 'Equalized Test Image')
        
    # # try a median filter to remove noise
    # lMedImage = exSki.filters.median(lTestImage)
    # fDisplayImageAndItsHistogram(lMedImage, 'Median Filtered Test Image')
    
    # # try a gaussian filter to lessen speckling impact
    lGaussSigma = 1
    # lTestGaussImage = exSki.filters.gaussian(lTestImage, sigma=lGaussSigma)
    # fDisplayImageAndItsHistogram(lTestGaussImage, f'Gaussian Filtered Test Image, sigma = {lGaussSigma}')
    
    # # try doing kmeans clustering to an image before and after gaussian filtering to compare
    # lKMeansTestImage = fDoKMeansClusteringOnImage(lTestImage, 2)
    # lKMeansTestGaussImage = fDoKMeansClusteringOnImage(lTestGaussImage, 2)
    # lFig, lAxes = plt.subplots(1,3, layout='tight')
    # lAxes[0].imshow(lTestImage, cmap='gray')
    # lAxes[0].set_title('Test Image')
    # lAxes[1].imshow(lKMeansTestImage, cmap='plasma')
    # lAxes[1].set_title('Clusters of Test Image')
    # lAxes[2].imshow(lKMeansTestGaussImage, cmap='plasma')
    # lAxes[2].set_title(f'Clusters of Gaussian Filtered\nTest Image (sigma={lTestImageGaussSigma})')
    # plt.show()
        
    #%% FIRST TEST FOR CLASSIFICATION: GAUSS FILTERING -> KMEANS CLUSTERING -> LOGISTIC REGRESSION

    # make the KMeans cluster set be a matrix where each column is the clustering of an image that has been gauss filtered
    lKMeansClusterSet = np.zeros((lM*lN, len(lImageSet)))
    for iImageIndex in range(len(lImageSet)):
        lGaussImage = exSki.filters.gaussian(lImageSet[iImageIndex], lGaussSigma) 
        lKMeansClusteredImage = fDoKMeansClusteringOnImage(lGaussImage, 2)
        lKMeansClusteredImage = lKMeansClusteredImage.reshape(-1, 1)
        lKMeansClusterSet[:, iImageIndex] = lKMeansClusteredImage[:, 0]
    
    lKMeansClusterSet = StandardScaler().fit_transform(lKMeansClusterSet)
    lKMeansClusterSet = lKMeansClusterSet.transpose() # make each row an image whose clusterings are the features
    
    lKMCTrainDataSet, lKMCTestDataSet, lKMCTrainClassSet, lKMCTestClassSet = train_test_split(lKMeansClusterSet, lImageSetClassifications, train_size=0.8)
    
    lGLCM_LRPredLabels, lGLCM_LRAccuracyScore = fDoLogisticRegressionClassification(lKMCTrainDataSet, lKMCTrainClassSet, lKMCTestDataSet, lKMCTestClassSet)
    
    # plot the predicted vs 'true' classifications
    
    
    #%% SECOND TEST FOR CLASSIFICATION: GRAY-LEVEL CO-OCCURRENCE MATRIX TEXTURE PROPERTIES -> LOGISTIC REGRESSION

    # setting the GLCM levels
    lMaxIntLevel = 0;        
    for iImage in lImageSet:
        lCurrentMaxInt = np.max(iImage)
        if lCurrentMaxInt > lMaxIntLevel: lMaxIntLevel = lCurrentMaxInt
    lGLCMlevels = lMaxIntLevel + 1
    
    # getting the GLCM properties for each image
    lGLCMSet = np.zeros((len(lImageSet), 4))  
    for iImageIndex in range(len(lImageSet)):
        lGLCM = exSki.feature.graycomatrix(lImageSet[iImageIndex], distances=[1], angles=[0], levels=lGLCMlevels)
        lGLCMSet[iImageIndex, 0] = exSki.feature.graycoprops(lGLCM, 'ASM')
        lGLCMSet[iImageIndex, 1] = exSki.feature.graycoprops(lGLCM, 'contrast')
        lGLCMSet[iImageIndex, 2] = exSki.feature.graycoprops(lGLCM, 'correlation')
        lGLCMSet[iImageIndex, 3] = exSki.feature.graycoprops(lGLCM, 'dissimilarity')
        
    lGLCMSet = StandardScaler().fit_transform(lGLCMSet)
        
    lGLCMTrainDataSet, lGLCMTestDataSet, lGLCMTrainClassSet, lGLCMTestClassSet = train_test_split(lGLCMSet, lImageSetClassifications, train_size=0.8)
    
    lGLCM_LRPredLabels, lGLCM_LRAccuracyScore = fDoLogisticRegressionClassification(lGLCMTrainDataSet, lGLCMTrainClassSet, lGLCMTestDataSet, lGLCMTestClassSet)
    
    # plot the predicted vs 'true' classifications
    
    
    
    



#%% FUNCTIONS

def fUpsizeImagesToLargestInSet(aImageSet): 
    lMaxImageRows, lMaxImageCols = 0, 0    
    for iImage in aImageSet: # find the largest image size in the set
        lCurrentImageRows, lCurrentImageCols = iImage.shape[0], iImage.shape[1]
        if iImage.shape[0] > lMaxImageRows: lMaxImageRows = iImage.shape[0]
        if iImage.shape[1] > lMaxImageCols: lMaxImageCols = iImage.shape[1]
    for iImageIndex in range(len(aImageSet)): # now that we have the max sizes, upsize everything to it
        aImageSet[iImageIndex] = exSki.exposure.rescale_intensity(aImageSet[iImageIndex], out_range = (0, 255)) # make them 8-bit to prevent memory problems
        aImageSet[iImageIndex] = exSki.transform.resize(aImageSet[iImageIndex], (lMaxImageRows, lMaxImageCols), preserve_range=True).astype(int)
    return aImageSet, lMaxImageRows, lMaxImageCols

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

def fDoLogisticRegressionClassification(aTrainX, aTrainY, aTestX, aTestY):
    lLRModel = LogisticRegression()
    lLRModel.fit(aTrainX, aTrainY)
    lPredictedLabels = lLRModel.predict(aTestX)
    return lPredictedLabels, accuracy_score(aTestY, lPredictedLabels)
    
if __name__=="__main__":
    main()


# lGLCMlevels = lMaxIntLevel + 1
# lGLCMSet = np.zeros((lGLCMlevels**2, len(lImageSet)))  
# for iImageIndex in range(len(lImageSet)):
#     # lGLCMlevels = np.max(lImageSet[iImageIndex])+1
#     lGLCM = exSki.feature.graycomatrix(lImageSet[iImageIndex], distances=[1], angles=[0], levels=lGLCMlevels)
#     lGLCM = lGLCM[:, :, 0, 0]
#     lGLCM = lGLCM.reshape(-1, 1)
#     lGLCMSet[0:lGLCMlevels**2, iImageIndex] = lGLCM[:, 0]