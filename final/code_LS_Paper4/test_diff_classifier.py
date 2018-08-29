# -*- coding: utf-8 -*-
#import getSeq3
#import encode_4
import algorithm_1
import logging
#import argparse
import numpy as np
#from sklearn.model_selection import LeaveOneOut
import tables
import h5filereader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
#from sklearn.model_selection import KFold
from datetime import datetime
startTime = datetime.now()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import tree

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
import logging
logging.basicConfig(level=logging.DEBUG)
if __name__ == '__main__':
    maxAcc =[]
    startTime = datetime.now()
    fileh = tables.open_file("encodingFile.h5", mode="r")        
    featureRowMatrix = fileh.root.featureRowMatrix
    
    labelVec = h5filereader.readLabelVecFromFile()
    numOfLabelData = len(labelVec)
    kFeatures = range(1, len(featureRowMatrix)+1)
    accuracyScoreList = []
    logging.info("Labeled data: %d",numOfLabelData)
    logging.info("Unlabeled data: %d", featureRowMatrix.shape[1] - numOfLabelData)
    logging.info("Classifier: KNN")
    logging.info("Cross-validation: leave-one-out")
#    loo = LeaveOneOut()
    loo = LeaveOneOut()
    kf = KFold(n_splits=5, shuffle=True)
    
    kFeatures = range(1,320)
    k_knn_graph = 85
    for k_KNN_feature_selection in range(k_knn_graph,k_knn_graph + 1): #k neightbor seq in knn graph
        print k_KNN_feature_selection
        logging.info("Number of neighbors in knn graph: %d", k_KNN_feature_selection)
        fileName = "knnGraph_"+str(k_KNN_feature_selection) + ".npy"
        sortedLaplaFeatureIndexes = np.array(algorithm_1.loadSortedLaplaFeatureIndexes(fileName))
        
#        sortedLaplaFeatureIndexes = tmp1
#        sortedFeatureRowMat = np.array([featureRowMatrix[index] for index in sortedLaplaFeatureIndexes])
        sortedFeatureRowMat = np.array(featureRowMatrix)[sortedLaplaFeatureIndexes]
        sumarizeAccuracy1 = []
        sumarizeAccuracy2 = []
        sumarizeAccuracy3 = []
        sumarizeAccuracy4 = []

        #Laplacian score
        for numOfFeature in kFeatures:
            logging.info("####################")
            logging.info("Number of selected features: %d", numOfFeature)
        
            print h5filereader.readFeatureListFromFile()[sortedLaplaFeatureIndexes[:numOfFeature]]
            # build seqRowMat
            seqRowMatrix = np.transpose(sortedFeatureRowMat[:numOfFeature])       
            seqRowOfLabelData = seqRowMatrix[:numOfLabelData]
            
#            KNN {'Accuracy': 79.200000000000003, 'numOfFeatures': 124}
            cv_result = cross_val_score(KNeighborsClassifier(n_neighbors=7, weights = 'distance'), seqRowOfLabelData, labelVec, cv=loo)
            print "KNN: \navg accuracy: " +  str(cv_result.mean())
            print "KNN: \navg accuracy: " +  str(cv_result)
            sumarizeAccuracy1.append(cv_result.mean())
            
#            #SVM
#            cv_result = cross_val_score(SVC(), seqRowOfLabelData, labelVec, cv=loo)
#            print "SVM: \navg accuracy: " +  str(cv_result.mean())
#            sumarizeAccuracy2.append(cv_result.mean())
##            
#            #Linear SVM
#            cv_result = cross_val_score(LinearSVC(penalty='l2', dual=False, C=50), seqRowOfLabelData, labelVec, cv=loo)
#            print "Linear SVM: \navg accuracy: " +  str(cv_result.mean())
#            sumarizeAccuracy3.append(cv_result.mean())
##            
###            
###            #tree
#            cv_result = cross_val_score(tree.DecisionTreeClassifier(), seqRowOfLabelData, labelVec, cv=loo)
#            print "tree: \navg accuracy: " +  str(cv_result.mean())
#            sumarizeAccuracy4.append(cv_result.mean())
#            
            # Naive Bayes
#            cv_result = cross_val_score(GaussianNB(), seqRowOfLabelData, labelVec, cv=loo)
#            print "Naive Bayes: \navg accuracy: " +  str(cv_result.mean())
#            sumarizeAccuracy3.append(cv_result.mean())
            
#            #AdaBoostClassifier
#            cv_result = cross_val_score(AdaBoostClassifier(), seqRowOfLabelData, labelVec, cv=loo)
#            print "AdaBoostClassifier: \navg accuracy: " +  str(cv_result.mean())
            
#            cv_result = cross_val_score(QuadraticDiscriminantAnalysis(), seqRowOfLabelData, labelVec, cv=loo)
#            print "QuadraticDiscriminantAnalysis: \navg accuracy: " +  str(cv_result.mean())
            
#            cv_result = cross_val_score(QuadraticDiscriminantAnalysis(), seqRowOfLabelData, labelVec, cv=loo)
#            print "QuadraticDiscriminantAnalysis: \navg accuracy: " +  str(cv_result.mean())
#            print "QuadraticDiscriminantAnalysis: \navg accuracy: " +  str(cv_result)
            
#            cv_result = cross_val_score(GaussianProcessClassifier(1.0 * RBF(1.0)), seqRowOfLabelData, labelVec, cv=loo)
#            print "GaussianProcessClassifier\navg accuracy: " +  str(cv_result.mean())
            
#            
#            #leave one out
#            
#            tmpAccuracyScoreList = []
#            
##            for train_index, test_index in kf.split(seqRowOfLabelData):   
#            for train_index, test_index in loo.split(seqRowOfLabelData):    
#                X_train, X_test = seqRowOfLabelData[train_index], seqRowOfLabelData[test_index]
#                y_train, y_test = labelVec[train_index], labelVec[test_index]
#                #KNN
#                neigh = KNeighborsClassifier(n_neighbors=5, metric = 'cosine', weights = 'distance')
#                neigh.fit(X_train, y_train)
#                labelPredict = neigh.predict(X_test)     
#                tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
#      
###                SVC          
##                clf = LinearSVC(penalty='l2', dual=False, C=50)
###                clf= SVC(kernel = 'linear', C=C)
##                clf.fit(X_train, y_train)
##                labelPredict = clf.predict(X_test)     
##                tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
#                
###                 SVC          
##                clf = SVC()
###                clf= SVC(kernel = 'linear', C=C)
##                clf.fit(X_train, y_train)
##                labelPredict = clf.predict(X_test)     
##                tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
#                
##                #Decision tree
##                clf = tree.DecisionTreeClassifier()
##                clf.fit(X_train, y_train)
##                labelPredict = clf.predict(X_test)     
##                tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
##                
##                #   Naive Bayes        
##                clf = GaussianNB()
##                clf.fit(X_train, y_train)
##                labelPredict = clf.predict(X_test)     
##                tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
#                
#                
##                #   AdaBoostClassifier        
##                clf = AdaBoostClassifier()
##                clf.fit(X_train, y_train)
##                labelPredict = clf.predict(X_test)     
##                tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
#                
##                #   neural_network       
##                clf = MLPClassifier(alpha=1)
##                clf.fit(X_train, y_train)
##                labelPredict = clf.predict(X_test)     
##                tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
#                
##                #   QuadraticDiscriminantAnalysis()       
##                clf = QuadraticDiscriminantAnalysis()
##                clf.fit(X_train, y_train)
##                labelPredict = clf.predict(X_test)     
##                tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
##                
##                #   GaussianProcessClassifier()       
##                clf = GaussianProcessClassifier(1.0 * RBF(1.0))
##                clf.fit(X_train, y_train)
##                labelPredict = clf.predict(X_test)     
##                tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
#                
#                    
#            sumarizeAccuracy.append(np.average(tmpAccuracyScoreList)*100)
#            logging.info("Accuracy: %f", np.average(tmpAccuracyScoreList)*100)
        
    fileh.close()
    
#    plt.plot(kFeatures, sumarizeAccuracy, "k-o")#, label="Laplacian score - KNN") #"g-o"
#    plt.ylabel(u'Độ chính xác (%)')
#    plt.xlabel(u'Số lượng thuộc tính được chọn')
#    #plt.yticks(range(0,101,10))
##        plt.xticks(np.arange(min(kFeatures), max(kFeatures)+1, 5))
##        plt.xticks(np.arange(280, 356, 5))
#    plt.legend()
#    plt.show()
        
    