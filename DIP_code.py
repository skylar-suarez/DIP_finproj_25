# -*- coding: utf-8 -*-
"""
BIOE 5100 Final Project - Spring 2025
Skylar Suarez & Larry Baker 

Variable naming convention: lLocalVariable, fUserDefinedFunction, exExternalLibrary, aArgument, iIterator
"""

import skimage as exSki 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

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
    
    # try histogram equalization for higher contrast
    # increasing contrast amplifies the speckling and makes kmeans worse
    lEqualizedImage = exSki.exposure.equalize_hist(lTestImage)
    fDisplayImageAndItsHistogram(lEqualizedImage, 'Equalized Test Image')
       
    # try a median filter to remove noise
    lMedImage = exSki.filters.median(lTestImage)
    fDisplayImageAndItsHistogram(lMedImage, 'Median Filtered Test Image')
    
    # try a gaussian filter to lessen speckling impact
    lGaussSigma = 1
    lTestGaussImage = exSki.filters.gaussian(lTestImage, sigma=lGaussSigma)
    fDisplayImageAndItsHistogram(lTestGaussImage, f'Gaussian Filtered Test Image, sigma = {lGaussSigma}')
    
    # try doing kmeans clustering to an image before and after gaussian filtering to compare
    lKMeansTestImage = fDoKMeansClusteringOnImage(lTestImage, 2)
    lKMeansTestGaussImage = fDoKMeansClusteringOnImage(lTestGaussImage, 2)
    lFig, lAxes = plt.subplots(1,3, layout='tight')
    lAxes[0].imshow(lTestImage, cmap='gray')
    lAxes[0].set_title('Test Image')
    lAxes[1].imshow(lKMeansTestImage, cmap='plasma')
    lAxes[1].set_title('Clusters of Test Image')
    lAxes[2].imshow(lKMeansTestGaussImage, cmap='plasma')
    lAxes[2].set_title(f'Clusters of Gaussian Filtered\nTest Image (sigma={lGaussSigma})')
    plt.show()
        
    #%% FIRST TEST FOR CLASSIFICATION: GAUSS FILTERING -> KMEANS CLUSTERING -> LOGISTIC REGRESSION and DECISION TREE 

    # make the KMeans cluster set be a matrix where each column is the clustering of an image that has been gauss filtered
    lKMeansClusterSet = np.zeros((lM*lN, len(lImageSet)))
    for iImageIndex in range(len(lImageSet)):
        lGaussImage = exSki.filters.gaussian(lImageSet[iImageIndex], lGaussSigma) 
        lKMeansClusteredImage = fDoKMeansClusteringOnImage(lGaussImage, 2)
        lKMeansClusteredImage = lKMeansClusteredImage.reshape(-1, 1)
        lKMeansClusterSet[:, iImageIndex] = lKMeansClusteredImage[:, 0]
    
    lKMeansClusterSet = StandardScaler().fit_transform(lKMeansClusterSet)
    lKMeansClusterSet = lKMeansClusterSet.transpose() # make each row an image whose clusterings are the features
    
    # split the KMeans Cluster Data into train and test sets
    lKMCTrainDataSet, lKMCTestDataSet, lKMCTrainClassSet, lKMCTestClassSet = train_test_split(lKMeansClusterSet, lImageSetClassifications, train_size=0.8)
    
    # do the Logistic Regression Classification with 5 fold cross validation
    lKMC_LRPredLabels, lKMC_LRPredictedProbabilities, lKMC_LRAccuracyScore = fDoCVLogisticRegressionClassification(lKMCTrainDataSet, lKMCTrainClassSet, lKMCTestDataSet, lKMCTestClassSet, 5)
    
    # plot the predicted vs true classifications
    fPlotPredictionsVsTruth(lKMC_LRPredictedProbabilities, lKMCTestClassSet, lKMC_LRAccuracyScore, "Logistic Regression on K-Means Clusters")
    
    # do the Decision Tree Classification with 5 fold cross validation
    lKMC_DTPredLabels, lKMC_DTPredictedProbabilities, lKMC_DTAccuracyScore = fDoCVDecisionTreeClassification(lKMCTrainDataSet, lKMCTrainClassSet, lKMCTestDataSet, lKMCTestClassSet, 5)
    
    # plot the predicted vs true classifications
    fPlotPredictionsVsTruth(lKMC_DTPredictedProbabilities, lKMCTestClassSet, lKMC_DTAccuracyScore, "Decision Tree on K-Means Clustering")
    
    
    #%% SECOND TEST FOR CLASSIFICATION: GRAY-LEVEL CO-OCCURRENCE MATRIX TEXTURE PROPERTIES -> LOGISTIC REGRESSION

    # setting the GLCM levels
    lMaxIntLevel = 0;        
    for iImage in lImageSet:
        lCurrentMaxInt = np.max(iImage)
        if lCurrentMaxInt > lMaxIntLevel: lMaxIntLevel = lCurrentMaxInt
    lGLCMlevels = lMaxIntLevel + 1
    
    # getting the GLCM properties for each image
    lGLCMSet = np.zeros((len(lImageSet), 5))  
    for iImageIndex in range(len(lImageSet)):
        lGLCM = exSki.feature.graycomatrix(lImageSet[iImageIndex], distances=[1], angles=[0], levels=lGLCMlevels)
        lGLCMSet[iImageIndex, 0] = exSki.feature.graycoprops(lGLCM, "ASM")
        lGLCMSet[iImageIndex, 1] = exSki.feature.graycoprops(lGLCM, "contrast")
        lGLCMSet[iImageIndex, 2] = exSki.feature.graycoprops(lGLCM, "correlation")
        lGLCMSet[iImageIndex, 3] = exSki.feature.graycoprops(lGLCM, "dissimilarity")
        lGLCMSet[iImageIndex, 4] = exSki.feature.graycoprops(lGLCM, "homogeneity")
    
    # standardize the GLCM data
    lGLCMSet = StandardScaler().fit_transform(lGLCMSet)
    
    # split the GLCM data into train and test sets
    lGLCMTrainDataSet, lGLCMTestDataSet, lGLCMTrainClassSet, lGLCMTestClassSet = train_test_split(lGLCMSet, lImageSetClassifications, train_size=0.8)
    
    # do the Logistic Regression Classification with 5 fold cross validation
    lGLCM_LRPredLabels, lGLCM_LRPredictedProbabilities, lGLCM_LRAccuracyScore = fDoCVLogisticRegressionClassification(lGLCMTrainDataSet, lGLCMTrainClassSet, lGLCMTestDataSet, lGLCMTestClassSet, 5)
    
    # plot the predicted vs true classifications
    fPlotPredictionsVsTruth(lGLCM_LRPredictedProbabilities, lGLCMTestClassSet, lGLCM_LRAccuracyScore, "Logistic Regression on GLCM")
    
    # do the Decision Tree Classification with 5 fold cross validation
    lGLCM_LRPredLabels, lGLCM_LRPredictedProbabilities, lGLCM_LRAccuracyScore = fDoCVDecisionTreeClassification(lGLCMTrainDataSet, lGLCMTrainClassSet, lGLCMTestDataSet, lGLCMTestClassSet, 5)
    
    # plot the predicted vs true classifications
    fPlotPredictionsVsTruth(lGLCM_LRPredictedProbabilities, lGLCMTestClassSet, lGLCM_LRAccuracyScore, "Decision Tree on GLCM")
    

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
    plt.plot(range(len(aLRPredictedProbabilities)), aTrueClassifications, '*', markersize=15, label='true classification')
    plt.title(f"{aDatasetName}, Test Set Prediction Performance\nAccuracy Score = {np.round(aLRAccuracyScore, 2)}")
    plt.hlines(y=0.5, xmin=0, xmax=10, colors='k')
    plt.xlabel("Test Image")
    plt.ylabel("Probability")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35))
    plt.show()
    
# def fDoPCAAnalysis(aDataFrame):
#     lPCAModel = PCA()
#     lPCAModel.fit_transform(aDataFrame)
#     lExplainedVarianceRatios = lPCAModel.explained_variance_ratio_ # list of % variance explained by each PC
#     lExplainedVarianceRatiosSum = np.cumsum(lExplainedVarianceRatios) # ordered cummulative sum of % variance explained by each PC
#     return lExplainedVarianceRatios, lExplainedVarianceRatiosSum, lPCAModel.components_

# def fPlotVarianceExplainedByPCs(aExplainedVarianceRAtios, aExplainedVarianceRatiosSum, aTitle):
#     # plot the PCs in descending order (higher explained variance first)
#     lFigure = plt.figure()
#     plt.bar(range(0,len(aExplainedVarianceRAtios)), aExplainedVarianceRAtios, alpha=0.5, align='center', label='Individual explained variance')
#     plt.step(range(0,len(aExplainedVarianceRatiosSum)), aExplainedVarianceRatiosSum, where='mid',label='Cumulative explained variance')
#     plt.ylabel('Explained variance ratio')
#     plt.xlabel('Principal component index')
#     plt.title(aTitle)
#     plt.legend(loc='best')
#     plt.tight_layout()
#     plt.show()
    
# def fPlotFeatureContributionToFirst2PCs(aComponentDF, aDatasetDescription):
#     lFig, lAxes = plt.subplots(ncols=2, nrows=1, layout='constrained', sharey=True)
#     lAxes[0].barh(aComponentDF.columns, aComponentDF.iloc[0], color='red', label='impact on 1st PC')
#     lAxes[1].barh(aComponentDF.columns, aComponentDF.iloc[1], color='blue', label='impact on 2nd PC')
#     lFig.suptitle(f'{aDatasetDescription} Dataset, 1st 2 PCs')
#     lFig.legend()
#     lFig.show()

    
if __name__=="__main__":
    main()

