# -*- coding: utf-8 -*-
import getSeq3
import encode_4
import algorithm_1
import logging
import argparse
import numpy as np
from sklearn.model_selection import LeaveOneOut
import tables
import h5filereader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from datetime import datetime
startTime = datetime.now()
#clf = LinearSVC(penalty='l2', dual=False, C=50)
logging.basicConfig(level=logging.DEBUG)
from sklearn import tree
#
from sklearn.svm import SVC


if __name__ == '__main__':
#    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='Laplacian score')
#    parser.add_argument('-nonsvr', help='non-svr isdr sequence files', dest='nr_isdr_files', action='append', required=True)
#    parser.add_argument('-svr', help='svr isdr sequence files', dest='r_isdr_files', action='append', required=True)
#    parser.add_argument('-nonlabel', help='non-labeled isdr sequence files', dest='nonlabel_isdr_files', action='append')
#    args = parser.parse_args()
#    nr_isdr_files = args.nr_isdr_files
#    r_isdr_files = args.r_isdr_files
#    nonlabel_isdr_files = args.nonlabel_isdr_files
    
    nr_isdr_files = ["./input2/nr_isdr.txt"]
    r_isdr_files = ["./input2/r_isdr.txt"]
    nonlabel_isdr_files = ["./input2/non_labeled_isdr.txt"]

    
    logging.info("------------------GET ISDR SEQUENCES FROM FILE------------------")
    getSeq3.getISDRSeq(nr_isdr_files, r_isdr_files, nonlabel_isdr_files)
    
    logging.info("------------------ENCODING------------------")
    encode_4.encode()
    
    logging.info("ENCODING")
    logging.info(datetime.now() - startTime)
    startTime = datetime.now()
    
    logging.info("------------------CALCULATE LAPLACIAN SCORE------------------")
    sizeOfSeqList = len(h5filereader.getSeqListFromH5File())
    k_knn_graph = sizeOfSeqList/50
    algorithm_1.calculateAllLaplacianScore(k_knn_graph=k_knn_graph)
    
    
    logging.info("------------------VISUALIZATION------------------")
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
    loo = LeaveOneOut()
#    kf = KFold(n_splits=5, shuffle=True)
    for k_KNN_feature_selection in range(k_knn_graph,k_knn_graph + 1): #k neightbor seq in knn graph
        print k_KNN_feature_selection
        logging.info("Number of neighbors in knn graph: %d", k_KNN_feature_selection)
        fileName = "knnGraph_"+str(k_KNN_feature_selection) + ".npy"
        sortedLaplaFeatureIndexes = np.array(algorithm_1.loadSortedLaplaFeatureIndexes(fileName))
#        sortedLaplaFeatureIndexes = tmp1
#        sortedFeatureRowMat = np.array([featureRowMatrix[index] for index in sortedLaplaFeatureIndexes])
        sortedFeatureRowMat = np.array(featureRowMatrix)[sortedLaplaFeatureIndexes]
        sumarizeAccuracy = []

        #Laplacian score
        for numOfFeature in kFeatures:
            logging.info("####################")
            logging.info("Number of selected features: %d", numOfFeature)
            # build seqRowMat
            seqRowMatrix = np.transpose(sortedFeatureRowMat[:numOfFeature])       
            seqRowOfLabelData = seqRowMatrix[:numOfLabelData]
            
            #leave one out
            
            tmpAccuracyScoreList = []
            
#            for train_index, test_index in kf.split(seqRowOfLabelData):   
            for train_index, test_index in loo.split(seqRowOfLabelData):    
                X_train, X_test = seqRowOfLabelData[train_index], seqRowOfLabelData[test_index]
                y_train, y_test = labelVec[train_index], labelVec[test_index]
                #KNN
                neigh = KNeighborsClassifier(n_neighbors=5, metric = 'cosine', weights = 'distance')
                neigh.fit(X_train, y_train)
                labelPredict = neigh.predict(X_test)     
                tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
                
#                #63.44 cosine
#                #SVC
#                clf = tree.DecisionTreeClassifier()                       
#                clf.fit(X_train, y_train)
#                labelPredict = clf.predict(X_test)     
#                tmpAccuracyScoreList.append(accuracy_score(y_test,labelPredict))
            sumarizeAccuracy.append(np.average(tmpAccuracyScoreList)*100)
            logging.info("Accuracy: %f", np.average(tmpAccuracyScoreList)*100)
        
        
    fileh.close()
#    np.save("semi_acc1", sumarizeAccuracy)
    sortAcc = [{"numOfFeatures": i[0]+1,"Accuracy": i[1]} for i in sorted(enumerate(sumarizeAccuracy),reverse=True, key=lambda x:x[1])]
    
    print "\n------------------------------FINAL RESULT------------------------------------"
    print 'TOP 10 ACCURACY'
    for item in sortAcc[:10]:
        print item
#        numOfFeature = item["numOfFeatures"]
#        
#        listIndexes = sortedLaplaFeatureIndexes[:numOfFeature]
#        print listIndexes
#        print h5filereader.readFeatureListFromFile()[listIndexes]
#        
    plt.plot(kFeatures, sumarizeAccuracy, "k-o")#, label="Laplacian score - KNN") #"g-o"
    plt.ylabel(u'Độ chính xác (%)')
    plt.xlabel(u'Số lượng thuộc tính được chọn')
    #plt.yticks(range(0,101,10))
#        plt.xticks(np.arange(min(kFeatures), max(kFeatures)+1, 5))
#        plt.xticks(np.arange(280, 356, 5))
    plt.legend()
    plt.show()
        
    logging.info("ALL PROCESS")
    logging.info(datetime.now() - startTime)


#-----------------------------------------------------------------------------
#Chi su dung nha
#kFeatures = range(1, 338)    
#a = np.load("supervised_acc.npy")    
#b = np.load("semi_acc.npy") 
#plt.plot(kFeatures, b[:337], "r-x", label=u"Học bán giám sát")
#plt.plot(kFeatures, a, "b-o", label=u"Học giám sát")
#plt.ylabel(u'Độ chính xác (%)')
#plt.xlabel(u'Số lượng thuộc tính được chọn')
##plt.yticks(range(0,101,10))
##        plt.xticks(np.arange(min(kFeatures), max(kFeatures)+1, 5))
##        plt.xticks(np.arange(280, 356, 5))
#plt.legend()
#plt.show()
#-----------------------------------------------------------------------------
#VARIANCE
#labelVec = h5filereader.readLabelVecFromFile()
#numOfLabelData = len(labelVec)
#
#
##variance
#seqRowMatrixFull = h5filereader.readSeqRowMatFromFile() 
#kFeatures = range(1, seqRowMatrixFull.shape[1]+1)
#sel = VarianceThreshold()
#sel.fit_transform(seqRowMatrixFull)
#sortedIndexVarList = np.flipud(np.argsort(sel.variances_))
#newFeatureRowMatrix = np.transpose(seqRowMatrixFull)[sortedIndexVarList]
#newSeqRowMatrix = np.transpose(newFeatureRowMatrix)
#newSeqRowOfLabelData = newSeqRowMatrix[:numOfLabelData]
#
#sumarizeAccuracy2 = []
#loo = LeaveOneOut()
##leave one out #variance 
#for numOfFeature in kFeatures:
#    print numOfFeature
#    seqRowOfLabelData = newSeqRowOfLabelData[:,:numOfFeature]
#    tmpAccuracyScoreList2 = []
#    for train_index, test_index in loo.split(seqRowOfLabelData):    
#        X_train, X_test = seqRowOfLabelData[train_index], seqRowOfLabelData[test_index]
#        y_train, y_test = labelVec[train_index], labelVec[test_index]
#        #KNN
#        neigh = KNeighborsClassifier(n_neighbors=5, metric = 'cosine', weights = 'distance')
#        neigh.fit(X_train, y_train)
#        labelPredict = neigh.predict(X_test)     
#        tmpAccuracyScoreList2.append(accuracy_score(y_test,labelPredict))
#    sumarizeAccuracy2.append(np.average(tmpAccuracyScoreList2))
#    
#np.save("variance_acc",sumarizeAccuracy2)


#kFeatures = range(1, 352)    
#a = np.load("variance_acc.npy")    
#b = np.load("semi_acc.npy") 
#plt.plot(kFeatures, b, "r-x", label=u"Độ do Laplacian")
#plt.plot(kFeatures, a*100, "b-o", label=u"Phương sai")
#plt.ylabel(u'Độ chính xác (%)')
#plt.xlabel(u'Số lượng thuộc tính được chọn')
##plt.yticks(range(0,101,10))
##        plt.xticks(np.arange(min(kFeatures), max(kFeatures)+1, 5))
##        plt.xticks(np.arange(280, 356, 5))
#plt.legend()
#plt.show()