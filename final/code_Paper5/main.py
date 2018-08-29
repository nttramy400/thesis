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
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
clf = LinearSVC(penalty='l2', dual=False, C=50)
logging.basicConfig(level=logging.DEBUG)
#
#if __name__ == '__main__':
#    parser = argparse.ArgumentParser(description='Laplacian score')
#    parser.add_argument('--nonsvr', help='FOO!')
##    parser.add_argument('-svr',  help='FOO!', required=True)
##    parser.add_argument('-unlabeled', help='FOO!', required=True)
#    parser.parse_args('--nonsvr')
#    #parser.add_argument('-unlabeled', type=int, default=42, help='FOO!', required=True, nargs=1)  
#
#    logging.info("------------------GET ISDR SEQUENCES FROM FILE------------------")
#    
#    logging.info("------------------ENCODING------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='Laplacian score')
    parser.add_argument('-nonsvr', help='non-svr isdr sequence files', dest='nr_isdr_files', action='append', required=True)
    parser.add_argument('-svr', help='svr isdr sequence files', dest='r_isdr_files', action='append', required=True)
    parser.add_argument('-nonlabel', help='non-labeled isdr sequence files', dest='nonlabel_isdr_files', action='append')

    args = parser.parse_args()
    logging.info("------------------GET ISDR SEQUENCES FROM FILE------------------")
    getSeq3.getISDRSeq(args.nr_isdr_files, args.r_isdr_files, args.nonlabel_isdr_files)
    
    logging.info("------------------ENCODING------------------")
    encode_4.encode()
    
    logging.info("------------------CALCULATE LAPLACIAN SCORE------------------")
    sizeOfSeqList = len(h5filereader.getSeqListFromH5File())
    k_knn_graph = sizeOfSeqList/50
    algorithm_1.calculateAllLaplacianScore(k_knn_graph=k_knn_graph)
    
    
    logging.info("------------------VISUALIZATION------------------")
    fileh = tables.open_file("encodingFile.h5", mode="r")        
    featureRowMatrix = fileh.root.featureRowMatrix
    
    labelVec = h5filereader.readLabelVecFromFile()
    numOfLabelData = len(labelVec)
    kFeatures = range(1, 50)
    accuracyScoreList = []
    
    loo = LeaveOneOut()
#    kf = KFold(n_splits=20)
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
            sumarizeAccuracy.append(np.average(tmpAccuracyScoreList)*100)
            logging.info("Accuracy: %f", np.average(tmpAccuracyScoreList)*100)
        
        
        plt.plot(kFeatures, sumarizeAccuracy, "g-o", label="Laplacian score - KNN")
        plt.ylabel('accuracy (%)')
        plt.xlabel('number of features')
        plt.yticks(range(0,101,10))
        plt.legend()
        plt.show()
        
    fileh.close()