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
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

if __name__ == '__main__':
    startTime = datetime.now()
    fileh = tables.open_file("../outputfile/encodingFile.h5", mode="r")        
    featureRowMatrix = fileh.root.featureRowMatrix
    sizeOfSeqList = featureRowMatrix.shape[1]
    labelDict = algFile.readLabelDictFromFile()
    labelVec = np.array(classifierFile.convertLabelDict2List(labelDict))
    numOfLabelData = len(labelVec)
    kFeatures = range(1, 200)
    accuracyScoreList = []
    accuracyScoreList2 = []
    sortedLaplaFeatureIndexes = algFile.loadSortedLaplaFeatureIndexes("3_3.npy")
    sortedFeatureRowMat = np.array([featureRowMatrix[index] for index in sortedLaplaFeatureIndexes])
    sumarizeAccuracy = []
    sumarizeAccuracy2 = []
    
    #variance
    seqRowMatrixFull = algFile.readSeqRowMatFromFile() 
#    sel = VarianceThreshold(threshold=(.6 * (1 - .6)))
    sel = VarianceThreshold(threshold=0.2)
    newSeqRowMatrix = sel.fit_transform(seqRowMatrixFull)
    newSeqRowOfLabelData = newSeqRowMatrix[:numOfLabelData]
    
    loo = LeaveOneOut()
    #Laplacian score
    for numOfFeature in kFeatures:
        print numOfFeature
        # build seqRowMat
        seqRowMatrix = np.transpose(sortedFeatureRowMat[:numOfFeature])        
        seqRowOfLabelData = seqRowMatrix[:numOfLabelData]
        
        #leave one out
        loo = LeaveOneOut()
        tmpAccuracyScoreList = []
        
        for train_index, test_index in loo.split(seqRowOfLabelData):    
            X_train, X_test = seqRowOfLabelData[train_index], seqRowOfLabelData[test_index]
            y_train, y_test = labelVec[train_index], labelVec[test_index]
            #KNN
            neigh = KNeighborsClassifier(n_neighbors=5, metric = 'cosine', weights = 'distance')
            neigh.fit(X_train, y_train)
            labelPredict = neigh.predict(X_test)     
            tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
        sumarizeAccuracy.append(np.average(tmpAccuracyScoreList))



        #leave one out #variance 
        seqRowOfLabelData = newSeqRowOfLabelData[:,:numOfFeature]
        tmpAccuracyScoreList2 = []
        for train_index, test_index in loo.split(seqRowOfLabelData):    
            X_train, X_test = seqRowOfLabelData[train_index], seqRowOfLabelData[test_index]
            y_train, y_test = labelVec[train_index], labelVec[test_index]
            #KNN
            neigh = KNeighborsClassifier(n_neighbors=5, metric = 'cosine', weights = 'distance')
            neigh.fit(X_train, y_train)
            labelPredict = neigh.predict(X_test)     
            tmpAccuracyScoreList2.append(accuracy_score(y_test,labelPredict))
        sumarizeAccuracy2.append(np.average(tmpAccuracyScoreList2))
       
        
        
        
    fileh.close()
    
    plt.plot(kFeatures, sumarizeAccuracy, "g-o", label="Laplacian score")
    plt.plot(kFeatures, sumarizeAccuracy2,"r-o", label="Variance")
    plt.ylabel('accuracy')
    plt.xlabel('number of features')
    plt.legend()
    plt.savefig("3_graphKNN3_modelKNN5.png")
    plt.show()

    print datetime.now() - startTime
    
    

    

