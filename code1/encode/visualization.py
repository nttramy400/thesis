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
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

version = "9"
testVer = "4"#1

if __name__ == '__main__':
    maxAcc =[]
    startTime = datetime.now()
    fileh = tables.open_file("..//outputfile//"+version+"//encodingFile.h5", mode="r")        
    featureRowMatrix = fileh.root.featureRowMatrix
    sizeOfSeqList = featureRowMatrix.shape[1]
    labelDict = algFile.readLabelDictFromFile()
    labelVec = np.array(classifierFile.convertLabelDict2List(labelDict))
    numOfLabelData = len(labelVec)
    kFeatures = range(1, 60)
    accuracyScoreList = []
    accuracyScoreList2 = []
    sumarizeAccuracy2 = []
    
    loo = LeaveOneOut()
    
    #variance
    seqRowMatrixFull = algFile.readSeqRowMatFromFile() 
    sel = VarianceThreshold()
    sel.fit_transform(seqRowMatrixFull)
    sortedIndexVarList = np.flipud(np.argsort(sel.variances_))
    newFeatureRowMatrix = np.transpose(seqRowMatrixFull)[sortedIndexVarList]
    newSeqRowMatrix = np.transpose(newFeatureRowMatrix)
    newSeqRowOfLabelData = newSeqRowMatrix[:numOfLabelData]
    
    #leave one out #variance 
    for numOfFeature in kFeatures:
        seqRowOfLabelData = newSeqRowOfLabelData[:,:numOfFeature]
        tmpAccuracyScoreList2 = []
        for train_index, test_index in loo.split(seqRowOfLabelData):    
            X_train, X_test = seqRowOfLabelData[train_index], seqRowOfLabelData[test_index]
            y_train, y_test = labelVec[train_index], labelVec[test_index]
            #KNN
            neigh = KNeighborsClassifier(n_neighbors=7, metric = 'cosine', weights = 'distance')
            neigh.fit(X_train, y_train)
            labelPredict = neigh.predict(X_test)     
            tmpAccuracyScoreList2.append(accuracy_score(y_test,labelPredict))
        sumarizeAccuracy2.append(np.average(tmpAccuracyScoreList2))
        
        
        
    for k_KNN_feature_selection in range(15,16): #k neightbor seq in knn graph
        print k_KNN_feature_selection
        fileName = "..//outputfile//"+version+ "//"+testVer+"//knnGraph_"+str(k_KNN_feature_selection) + ".npy"
        sortedLaplaFeatureIndexes = np.array(algFile.loadSortedLaplaFeatureIndexes(fileName))
#        sortedLaplaFeatureIndexes = tmp1
#        sortedFeatureRowMat = np.array([featureRowMatrix[index] for index in sortedLaplaFeatureIndexes])
        sortedFeatureRowMat = np.array(featureRowMatrix)[sortedLaplaFeatureIndexes]
        sumarizeAccuracy = []

        #Laplacian score
        for numOfFeature in kFeatures:
            
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
            
            for train_index, test_index in loo.split(seqRowOfLabelData):    
                X_train, X_test = seqRowOfLabelData[train_index], seqRowOfLabelData[test_index]
                y_train, y_test = labelVec[train_index], labelVec[test_index]
                
                #KNN
                neigh = KNeighborsClassifier(n_neighbors=7, metric = 'cosine', weights = 'distance')
                neigh.fit(X_train, y_train)
                labelPredict = neigh.predict(X_test)     
                tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
            sumarizeAccuracy.append(np.average(tmpAccuracyScoreList))
    
    
    
        plt.plot(kFeatures, sumarizeAccuracy, "g-o", label="Laplacian score")
        plt.plot(kFeatures, sumarizeAccuracy2,"r-o", label="Variance")
        plt.ylabel('accuracy')
        plt.xlabel('number of features')
        plt.legend()
        plt.show()
#        plt.savefig("..//outputfile//"+version+ "//"+testVer+"//knnGraph_"+str(k_KNN_feature_selection)+"-knnClassifier_7.png")
#        plt.close()
        
        
        maxAcc.append([k_KNN_feature_selection, max(sumarizeAccuracy)])
       
        
    fileh.close()
    print datetime.now() - startTime
    


    
#    plt.plot(kFeatures, sumarizeAccuracy,'r-o',kFeatures, sumarizeAccuracy2,"g-o")
#    plt.ylabel('accuracy')
#    plt.xlabel('number of features')
#    plt.show()
    
    
    
#    filter(lambda tup: tup[1] == max(sumarizeAccuracy), enumerate(sumarizeAccuracy))
    
import numpy as np
arrIndexList =  []
k_knn_graph = range(1, 200)
for i in k_knn_graph:
    featureList = np.load("D:/git_thesis/code1/outputfile/10/1/knnGraph_"+ str(i)+".npy")
    arrIndexList.append(list(featureList))
    
import collections   
def getKBestFeatureIndex(k, listIndexes):
    tmpKBestFeature = np.array(listIndexes)[:,:k]
    tmpKBestFeature = tmpKBestFeature.reshape(-1)
    featureFreqDict =collections.Counter(list(tmpKBestFeature))
#    featureFreqDict = {}
#    for row in tmpKBestFeature:
#        for curFeature in row:
#            if curFeature in featureFreqDict: #count so lan xuat hien sub subStr
#                featureFreqDict[curFeature] += 1
#            else:
#                featureFreqDict[curFeature] = 1
#                
#    return featureFreqDict
    
    #get k feature has largest value
    return [key for key in sorted(featureFreqDict, key=featureFreqDict.get, reverse=True)[:k]]

tmp1 = getKBestFeatureIndex(175,arrIndexList)
    

