import algorithm as algFile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
#def convertLabelDict2ListNum(labelDict):
#    labelToNum = 0 
#    listOfTupleLabels = []
#    numOfLabelData = algFile.getLastIndexLabel(labelDict)+1
#    labelVec = [None]*numOfLabelData
#    for label, listIndexSeq in labelDict.iteritems():
#        listOfTupleLabels.append((label,labelToNum))
#        for index in listIndexSeq:
#            labelVec[index] = labelToNum  
#        labelToNum += 1
#    return (listOfTupleLabels, labelVec)  

def convertLabelDict2List(labelDict):
    numOfLabelData = algFile.getLastIndexLabel(labelDict)+1
    labelVec = [None]*numOfLabelData
    for label, listIndexSeq in labelDict.iteritems():
        for index in listIndexSeq:
            labelVec[index] = label  
    return labelVec

if __name__ == '__main__':
    labelDict = algFile.readLabelDictFromFile()
    labelVec = convertLabelDict2List(labelDict)
#    labelVec.remove(None)
    numOfLabelData = len(labelVec)

    seqRowMatrix = algFile.readSeqRowMatFromFile('../outputfile/afterFeatureSelection.h5') 
    seqRowOfLabelData = seqRowMatrix[:numOfLabelData]
    

    #KNN
    neigh = KNeighborsClassifier(n_neighbors=11, metric = 'cosine', weights = 'distance')
    neigh.fit(seqRowOfLabelData, labelVec)
    labelPredict = neigh.predict(seqRowOfLabelData)
    accuracy_score(labelVec,labelPredict)

#    seqRowMatrix = algFile.readSeqRowMatFromFile() 
#    clf = svm.SVC()
#    clf.fit(seqRowMatrix[:numOfLabelData], labelVec)
    
#    labelReal = labelVec[77]
    
    seqRowMatrixFull = algFile.readSeqRowMatFromFile() 
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=(.7 * (1 - .7)))
    newSeqRowMatrix = sel.fit_transform(seqRowMatrixFull)
    newSeqRowOfLabelData = newSeqRowMatrix[:numOfLabelData]
    neigh = KNeighborsClassifier(n_neighbors=11, metric = 'cosine', weights = 'distance')
    neigh.fit(newSeqRowOfLabelData, labelVec)
    neigh.predict(newSeqRowOfLabelData)
    accuracy_score(labelVec,neigh.predict(newSeqRowOfLabelData))