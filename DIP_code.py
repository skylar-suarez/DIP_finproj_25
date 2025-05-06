# -*- coding: utf-8 -*-
"""
BIOE 5100 Final Project - Spring 2025
Skylar Suarez & Larry Baker 

Variable naming convention: lLocalVariable, fUserDefinedFunction, exExternalLibrary, aArgument, iIterator
"""
#%% LIBRARY IMPORTS

import skimage as exSki # For basic image processing (filtering, histogram equalization) and GLCM analysis.

import numpy as np # For matrix manipulation.
from numpy import fft # For frequency filtering.

import matplotlib.pyplot as plt # For plotting figures throughout.

from sklearn.cluster import KMeans # For implementing K-Means clustering on the images before logistic regression.
from sklearn.model_selection import train_test_split # For linear regression testing.
from sklearn.preprocessing import StandardScaler # For use with the GLCM parameters.
from sklearn.linear_model import LogisticRegression # For linear regression testing.
from sklearn.tree import DecisionTreeClassifier # For decision tree classification testing.
from sklearn.model_selection import cross_val_score # For linear regression testing.
from sklearn.metrics import accuracy_score # For evaluationg trained regression models.

#%% 
def main():
    # Read in all tif images into a list.
    lImageSet = list(exSki.io.imread_collection('images/*.tif', {'as_gray': True}))
    
    # Show the entire collection for presentation purposes.
    # exSki.io.imshow_collection(lImageSet, plugin='matplotlib', cmap='gray')
    # exSki.io.show()
        
    lImageSet, lM, lN = fResizeImagesInSet(lImageSet, 'min')
    
    # Make the vector of whether the images have AB or not (1 = AB, 0 = no AB).
    lImageSetClassifications = np.ones(len(lImageSet), dtype=int)
    lImageSetClassifications[[5,6,13,14,21,22,28,29,33,34,36,37,38,39,41,42,44,45,46,47,48]] = 0
    
    # Display which images are AB and which are not.
    plt.figure(figsize=(15, 1.5))
    plt.plot(range(len(lImageSet)), lImageSetClassifications, 'o')
    plt.xticks(range(len(lImageSet)))
    plt.yticks([0,1])
    plt.title('1 = AB, 0 = no AB')
    plt.show()
    
    #%% TESTING STUFF ON A SAMPLE IMAGE, for presentation purposes.
    lSampleImage = lImageSet[15]
    
    # Try histogram equalization for higher contrast -
    # Increasing contrast amplifies the speckling and makes kmeans worse.
    lEqualizedImage = exSki.exposure.equalize_hist(lSampleImage)
    fDisplayImageAndItsHistogram(lEqualizedImage, 'Equalized Sample Image')
    
    # Trying frequency filtering of the sample image.
    # Outputting frequency spectrum of image(s) to determine if frequency filtering is sensible. 
    lTestFFTOutput = np.abs(fft.fftshift(fft.fft2(lSampleImage))) # fft2() is 2D FFT, fftshift() used for same rationale as in MATLAB.
    # exSki.io.imshow(np.log(testFFTOutput), cmap = 'Blues') # Log needs to be used here or else fft is uninterpretable.
    # exSki.io.show()
    
    # From the above, frequency filtering may not be useful due to no clear pattern present in the spectrum.
    # An attempted filtering with a high-pass built-in Butterworth filter was attempted.
    lFreqFilteredImage = exSki.filters.butterworth(lSampleImage, cutoff_frequency_ratio = 0.1, high_pass = False) 
    # cutoff_frequency_ratio sets the cut-off freq. relative to FFT shape (i.e., relative to the whole sampled freq. range).
    lFigure, lAxes = plt.subplots(1, 3)
    lFigure.suptitle('Frequency Spectrum and Filtering with LPF on Sample Image')
    lAxes[0].set_title('Freq Spectrum')
    lAxes[0].imshow(np.log(lTestFFTOutput), cmap = 'Blues')
    lAxes[1].set_title('Original')
    lAxes[1].imshow(lSampleImage, cmap = 'gray')
    lAxes[2].set_title('Filtered')
    lAxes[2].imshow(lFreqFilteredImage, cmap = 'gray')
    plt.show()
    # From the above, the results of filtering present a blurred image that appears superficially similar to a Gaussian-blurred image. 
    
    # Trying Laplacian of Gaussian on the frequency-filtered image to see which 'blob' regions would be detected by the method 
    # (i.e., whether the plaque regions would stand out relative to the background by having clusters of identified blobs).
    lBlobLog = exSki.feature.blob_log(lFreqFilteredImage, min_sigma = 1, max_sigma = 20, num_sigma = 20, threshold_rel = 0.4)
    lFigure, lAxes = plt.subplots(1,2, layout = 'tight')
    lFigure.suptitle('LoG Blob Detection with a Frequency-Filtered Sample Image')
    lAxes[0].set_title('Original')
    lAxes[0].imshow(lSampleImage, cmap = 'gray')
    lAxes[1].set_title('Freq-Filtered')
    lAxes[1].imshow(lFreqFilteredImage, cmap = 'gray')
    for blob in lBlobLog:
            y, x, r = blob
            c_orig = plt.Circle((x, y), r, color = 'red', linewidth=1, fill=False)
            lAxes[0].add_patch(c_orig)
            c_freq = plt.Circle((x, y), r, color = 'red', linewidth=1, fill=False)
            lAxes[1].add_patch(c_freq)
       
    # Try a median filter to remove noise.
    lMedImage = exSki.filters.median(lSampleImage)
    fDisplayImageAndItsHistogram(lMedImage, 'Median Filtered Sample Image')
    
    # Try a Gaussian filter to lessen speckling impact.
    lGaussSigma = 1
    lTestGaussImage = exSki.filters.gaussian(lSampleImage, sigma=lGaussSigma)
    fDisplayImageAndItsHistogram(lTestGaussImage, f'Gaussian Filtered Test Image, sigma = {lGaussSigma}')
    
    # Trying the Canny edge-detection algorithm. Trying on the freqency-filtered version of the sample image.
    imageEdges = exSki.feature.canny(lFreqFilteredImage, sigma = 8, low_threshold = 0.8, high_threshold = 1)
    lFigure, lAxes = plt.subplots(1, 2, layout = 'compressed')
    lFigure.suptitle('Canny Edge Detection on the Sample Image')
    lAxes[0].imshow(lSampleImage, cmap = 'gray')
    lAxes[0].set_title('Original')
    lAxes[1].imshow(imageEdges, cmap = 'gray')
    lAxes[1].set_title('Edges Detected by Canny Edge Detection')
    plt.show() 
    # From the above, Canny edge detection does not appear to be an appropriate means of detecting areas of plaque.
    # If the histogram-equalized image is used, a large number of edges are detected at various threshold and sigma values, whereas
    # if the original is used, even more spurious edges are detected around pixels of noise. 
    # The Gaussian similarly does not provide adequate results, as would be expected since Canny edge detection
    # already performs a Gaussian blur to start.
    
    # Try doing kmeans clustering to an image before and after low pass frequency filtering to compare.
    lKMeansTestImage = fDoKMeansClusteringOnImage(lSampleImage, 2)
    lKMeansTestGaussImage = fDoKMeansClusteringOnImage(lFreqFilteredImage, 2)
    lFig, lAxes = plt.subplots(1,3, layout='tight')
    lAxes[0].imshow(lSampleImage, cmap='gray')
    lAxes[0].set_title('Sample Image')
    lAxes[1].imshow(lKMeansTestImage, cmap='plasma')
    lAxes[1].set_title('Clusters of Sample Image')
    lAxes[2].imshow(lKMeansTestGaussImage, cmap='plasma')
    lAxes[2].set_title(f'Clusters of Low Pass\nFiltered Sample Image)')
    plt.show()
        
    #%% FIRST TEST FOR CLASSIFICATION: GAUSS/LPF FILTERING -> KMEANS CLUSTERING -> LOGISTIC REGRESSION and DECISION TREE 
    
    lNumOfKFoldsForCV = 10

    # Make the KMeans cluster set be a matrix where each column is the clustering of an image that has been Gaussian filtered.
    lKMeansClusterSet = np.zeros((lM*lN, len(lImageSet)))
    for iImageIndex in range(len(lImageSet)):
        # lGaussImage = exSki.filters.gaussian(lImageSet[iImageIndex], lGaussSigma)
        lFreqFilteredImage = exSki.filters.butterworth(lImageSet[iImageIndex], cutoff_frequency_ratio = 0.1, high_pass = False)
        lKMeansClusteredImage = fDoKMeansClusteringOnImage(lFreqFilteredImage, 2)
        lKMeansClusteredImage = lKMeansClusteredImage.reshape(-1, 1)
        lKMeansClusterSet[:, iImageIndex] = lKMeansClusteredImage[:, 0]
    
    lKMeansClusterSet = StandardScaler().fit_transform(lKMeansClusterSet)
    lKMeansClusterSet = lKMeansClusterSet.transpose() # Make each row an image whose clusterings are the features.
    
    # Split the KMeans Cluster Data into train and test sets.
    lKMCTrainDataSet, lKMCTestDataSet, lKMCTrainClassSet, lKMCTestClassSet = train_test_split(lKMeansClusterSet, lImageSetClassifications, train_size=0.8)
    
    # Do the Logistic Regression Classification with 5 fold cross validation.
    lKMC_LRPredLabels, lKMC_LRPredictedProbabilities, lKMC_LRAccuracyScore = fDoCVLogisticRegressionClassification(lKMCTrainDataSet, lKMCTrainClassSet, lKMCTestDataSet, lKMCTestClassSet, lNumOfKFoldsForCV)
    
    # Plot the predicted vs true classifications.
    fPlotPredictionsVsTruth(lKMC_LRPredictedProbabilities, lKMCTestClassSet, lKMC_LRAccuracyScore, "Logistic Regression on K-Means Clusters")
    
    # Do the Decision Tree Classification with 5 fold cross validation.
    lKMC_DTPredLabels, lKMC_DTPredictedProbabilities, lKMC_DTAccuracyScore = fDoCVDecisionTreeClassification(lKMCTrainDataSet, lKMCTrainClassSet, lKMCTestDataSet, lKMCTestClassSet, lNumOfKFoldsForCV)
    
    # Plot the predicted vs true classifications.
    fPlotPredictionsVsTruth(lKMC_DTPredictedProbabilities, lKMCTestClassSet, lKMC_DTAccuracyScore, "Decision Tree on K-Means Clustering")
    
    
    #%% SECOND TEST FOR CLASSIFICATION: GRAY-LEVEL CO-OCCURRENCE MATRIX TEXTURE PROPERTIES -> LOGISTIC REGRESSION and DECISION TREE

    # Setting the GLCM levels.
    lMaxIntLevel = 0;        
    for iImage in lImageSet:
        lCurrentMaxInt = np.max(iImage)
        if lCurrentMaxInt > lMaxIntLevel: lMaxIntLevel = lCurrentMaxInt
    lGLCMlevels = lMaxIntLevel + 1
    
    # Getting the GLCM properties for each image.
    lGLCMSet = np.zeros((len(lImageSet), 5))  
    for iImageIndex in range(len(lImageSet)):
        lGLCM = exSki.feature.graycomatrix(lImageSet[iImageIndex], distances=[1], angles=[0], levels=lGLCMlevels)
        lGLCMSet[iImageIndex, 0] = exSki.feature.graycoprops(lGLCM, "ASM")
        lGLCMSet[iImageIndex, 1] = exSki.feature.graycoprops(lGLCM, "contrast")
        lGLCMSet[iImageIndex, 2] = exSki.feature.graycoprops(lGLCM, "correlation")
        lGLCMSet[iImageIndex, 3] = exSki.feature.graycoprops(lGLCM, "dissimilarity")
        lGLCMSet[iImageIndex, 4] = exSki.feature.graycoprops(lGLCM, "homogeneity")
    
    # Standardize the GLCM data.
    lGLCMSet = StandardScaler().fit_transform(lGLCMSet)
    
    # Split the GLCM data into train and test sets.
    lGLCMTrainDataSet, lGLCMTestDataSet, lGLCMTrainClassSet, lGLCMTestClassSet = train_test_split(lGLCMSet, lImageSetClassifications, train_size=0.8)
    
    # Do the Logistic Regression Classification with 5 fold cross validation.
    lGLCM_LRPredLabels, lGLCM_LRPredictedProbabilities, lGLCM_LRAccuracyScore = fDoCVLogisticRegressionClassification(lGLCMTrainDataSet, lGLCMTrainClassSet, lGLCMTestDataSet, lGLCMTestClassSet, lNumOfKFoldsForCV)
    
    # Plot the predicted vs true classifications.
    fPlotPredictionsVsTruth(lGLCM_LRPredictedProbabilities, lGLCMTestClassSet, lGLCM_LRAccuracyScore, "Logistic Regression on GLCM")
    
    # Do the Decision Tree Classification with 5 fold cross validation.
    lGLCM_DTPredLabels, lGLCM_DTPredictedProbabilities, lGLCM_DTAccuracyScore = fDoCVDecisionTreeClassification(lGLCMTrainDataSet, lGLCMTrainClassSet, lGLCMTestDataSet, lGLCMTestClassSet, lNumOfKFoldsForCV)
    
    # Plot the predicted vs true classifications.
    fPlotPredictionsVsTruth(lGLCM_DTPredictedProbabilities, lGLCMTestClassSet, lGLCM_DTAccuracyScore, "Decision Tree on GLCM")
    

#%% FUNCTIONS

def fResizeImagesInSet(aImageSet, mode='min'): 
    if mode=='max':
        lTargetImageRows, lTargetImageCols = 0, 0    
        for iImage in aImageSet: # find the largest image size in the set
            if iImage.shape[0] > lTargetImageRows: lTargetImageRows = iImage.shape[0]
            if iImage.shape[1] > lTargetImageCols: lTargetImageCols = iImage.shape[1]
    else:
        lTargetImageRows, lTargetImageCols = 100000, 100000    
        for iImage in aImageSet: # find the largest image size in the set
            if iImage.shape[0] < lTargetImageRows: lTargetImageRows = iImage.shape[0]
            if iImage.shape[1] < lTargetImageCols: lTargetImageCols = iImage.shape[1]     
    for iImageIndex in range(len(aImageSet)): # now that we have the target sizes, resize everything to it
        aImageSet[iImageIndex] = exSki.exposure.rescale_intensity(aImageSet[iImageIndex], out_range = (0, 255)) # make them 8-bit to prevent memory problems
        aImageSet[iImageIndex] = exSki.transform.resize(aImageSet[iImageIndex], (lTargetImageRows, lTargetImageCols), preserve_range=True).astype(int)
    return aImageSet, lTargetImageRows, lTargetImageCols

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
    lKMeansModel = KMeans(n_clusters=aK)
    lKMeansModel.fit(lImage)
    return lKMeansModel.predict(lImage).reshape(lImageShape)

def fDoCVLogisticRegressionClassification(aTrainX, aTrainY, aTestX, aTestY, aCV):
    lLRModel = LogisticRegression()
    lLRModel.fit(aTrainX, aTrainY)
    lPredictedLabels = lLRModel.predict(aTestX)
    lPredictedProbabilities = lLRModel.predict_proba(aTestX)
    lCV_RMSEs = cross_val_score(lLRModel, aTrainX, aTrainY, scoring='neg_mean_squared_error', cv=aCV)
    print(f'Logistic Regression Mean Squared Errors, CV of {aCV} folds = {lCV_RMSEs}')
    print(f'Average CV error = {np.round(np.sqrt(-np.mean(lCV_RMSEs)), 2)}')
    return lPredictedLabels, lPredictedProbabilities, accuracy_score(aTestY, lPredictedLabels)

def fDoCVDecisionTreeClassification(aTrainX, aTrainY, aTestX, aTestY, aCV):
    lDTModel = DecisionTreeClassifier()
    lDTModel.fit(aTrainX, aTrainY)
    lPredictedLabels = lDTModel.predict(aTestX)
    lPredictedProbabilities = lDTModel.predict_proba(aTestX)
    lCV_RMSEs = cross_val_score(lDTModel, aTrainX, aTrainY, scoring='neg_mean_squared_error', cv=aCV)
    print(f'Decision Tree Mean Squared Errors, CV of {aCV} folds = {lCV_RMSEs}')
    print(f'Average CV error = {np.round(np.sqrt(-np.mean(lCV_RMSEs)), 2)}')
    return lPredictedLabels, lPredictedProbabilities, accuracy_score(aTestY, lPredictedLabels)

def fPlotPredictionsVsTruth(aLRPredictedProbabilities, aTrueClassifications, aLRAccuracyScore, aDatasetName):
    plt.figure()
    plt.plot(range(len(aLRPredictedProbabilities)), aLRPredictedProbabilities[:,0], 'o', label='probability of class 0 membership')
    plt.plot(range(len(aLRPredictedProbabilities)), aLRPredictedProbabilities[:,1], 'o', label='probability class 1 membership')
    plt.plot(range(len(aLRPredictedProbabilities)), aTrueClassifications, '*', fillstyle='none', markersize=20, label='true classification')
    plt.title(f"{aDatasetName}, Test Set Prediction Performance\nAccuracy Score = {np.round(aLRAccuracyScore, 2)}")
    plt.hlines(y=0.5, xmin=0, xmax=10, colors='k')
    plt.xlabel("Test Image")
    plt.ylabel("Probability")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35))
    plt.show()
    
    
if __name__=="__main__":
    main()

