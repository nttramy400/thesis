import algorithm as algFile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics.pairwise import cosine_distances
#def convertLabelDict2List2(labelDict):
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
#    tmp = convertLabelDict2List(labelDict)
#    tmp[1].remove(None)
#    labelVec = tmp[1]
    labelVec = convertLabelDict2List(labelDict)
    labelVec.remove(None)
    numOfLabelData = len(labelVec)
    seqRowMatrix = algFile.readSeqRowMatFromFile('../outputfile/afterFeatureSelection.h5') 
    seqRowOfLabelData = seqRowMatrix[:numOfLabelData]
    
    neigh = KNeighborsClassifier(n_neighbors=5, metric=cosine_distances)
    tmp2 = neigh.fit(seqRowOfLabelData, labelVec)
    labelPredict = neigh.predict(seqRowOfLabelData)
    
    clf = svm.SVC()
    clf.fit(seqRowOfLabelData, labelVec)
    
#    seqRowMatrix = algFile.readSeqRowMatFromFile() 
#    clf = svm.SVC()
#    clf.fit(seqRowMatrix[:numOfLabelData], labelVec)
    
#    labelReal = labelVec[77]