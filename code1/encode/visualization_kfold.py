import algorithm as algFile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
import classifier as classifierFile
import tables
from datetime import datetime
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

version = "3"
testVer = "1"

if __name__ == '__main__':
    startTime = datetime.now()
    fileh = tables.open_file("..//outputfile//"+version+"//encodingFile.h5", mode="r")        
    featureRowMatrix = fileh.root.featureRowMatrix
    sizeOfSeqList = featureRowMatrix.shape[1]
    labelDict = algFile.readLabelDictFromFile()
    labelVec = np.array(classifierFile.convertLabelDict2List(labelDict))
    numOfLabelData = len(labelVec)
    kFeatures = range(1, 150)
    accuracyScoreList = []
    accuracyScoreList2 = []
    sumarizeAccuracy2 = []
    
    
    
    kf = KFold(n_splits=20)
    #variance
    seqRowMatrixFull = algFile.readSeqRowMatFromFile() 
    sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    newSeqRowMatrix = sel.fit_transform(seqRowMatrixFull)
    newSeqRowOfLabelData = newSeqRowMatrix[:numOfLabelData]
    
    #leave one out #variance 
    for numOfFeature in kFeatures:
        seqRowOfLabelData = newSeqRowOfLabelData[:,:numOfFeature]
        tmpAccuracyScoreList2 = []
        for train_index, test_index in kf.split(seqRowOfLabelData):    
            X_train, X_test = seqRowOfLabelData[train_index], seqRowOfLabelData[test_index]
            y_train, y_test = labelVec[train_index], labelVec[test_index]
            #KNN
            neigh = KNeighborsClassifier(n_neighbors=9, metric = 'cosine', weights = 'distance')
            neigh.fit(X_train, y_train)
            labelPredict = neigh.predict(X_test)     
            tmpAccuracyScoreList2.append(accuracy_score(y_test,labelPredict))
        sumarizeAccuracy2.append(np.average(tmpAccuracyScoreList2))
        
        
        
    for k_KNN_feature_selection in range(1,50):
        fileName = "..//outputfile//"+version+ "//"+testVer+"//knnGraph_"+str(k_KNN_feature_selection) + ".npy"
        sortedLaplaFeatureIndexes = algFile.loadSortedLaplaFeatureIndexes(fileName)
        sortedFeatureRowMat = np.array([featureRowMatrix[index] for index in sortedLaplaFeatureIndexes])
#        sortedFeatureRowMat = featureRowMatrix[sortedLaplaFeatureIndexes]
        sumarizeAccuracy = []

        #Laplacian score
        for numOfFeature in kFeatures:
            print numOfFeature
            # build seqRowMat
            seqRowMatrix = np.transpose(sortedFeatureRowMat[:numOfFeature])       
            seqRowOfLabelData = seqRowMatrix[:numOfLabelData]
            
    #        #build classifier
    #        neigh = KNeighborsClassifier(n_neighbors=11, metric = 'cosine', weights = 'distance')
    #        neigh.fit(seqRowOfLabelData, labelVec)
    #        labelPredict = neigh.predict(seqRowOfLabelData)
    #        accuracyScoreList.append(accuracy_score(labelVec,labelPredict))
            
            #leave one out
            
            tmpAccuracyScoreList = []
            
            for train_index, test_index in kf.split(seqRowOfLabelData):    
                X_train, X_test = seqRowOfLabelData[train_index], seqRowOfLabelData[test_index]
                y_train, y_test = labelVec[train_index], labelVec[test_index]
                
                #KNN
                neigh = KNeighborsClassifier(n_neighbors=9, metric = 'cosine', weights = 'distance')
                neigh.fit(X_train, y_train)
                labelPredict = neigh.predict(X_test)     
                tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
            sumarizeAccuracy.append(np.average(tmpAccuracyScoreList))
    
    
    
        plt.plot(kFeatures, sumarizeAccuracy, "g-o", label="Laplacian score")
        plt.plot(kFeatures, sumarizeAccuracy2,"r-o", label="Variance")
        plt.ylabel('accuracy')
        plt.xlabel('number of features')
        plt.legend()
        plt.savefig("..//outputfile//"+version+ "//"+testVer+"//knnGraph_"+str(k_KNN_feature_selection)+"-knnClassifier_9.png")
        plt.close()
       
        
    fileh.close()
    print datetime.now() - startTime
    


    
#    plt.plot(kFeatures, sumarizeAccuracy,'r-o',kFeatures, sumarizeAccuracy2,"g-o")
#    plt.ylabel('accuracy')
#    plt.xlabel('number of features')
#    plt.show()
    

