import numpy as np
import tables
from scipy.spatial.distance import cosine


############################ READ INFO FROM FILE ############################
def readFeatureRowMatFromFile(filename="../outputfile/encodingFile.h5"):
    fileh = tables.open_file(filename, mode="r")        
    featureRowMatrix = fileh.root.featureRowMatrix.read()
    fileh.close()
    return featureRowMatrix
    
def readSeqRowMatFromFile(filename="../outputfile/encodingFile.h5"):
    fileh = tables.open_file(filename, mode="r")        
    seqRowMatrix = fileh.root.seqRowMatrix.read()
    fileh.close()
    return seqRowMatrix

def readFeatureListFromFile(filename="../outputfile/encodingFile.h5"):
    fileh = tables.open_file(filename, mode="r")        
    seqRowMatrix = fileh.root.feature.read()
    fileh.close()
    return seqRowMatrix


def readLabelDictFromFile(filename="../outputfile/info.h5"):
    fileh = tables.open_file(filename, mode="r")
    labelDict = {}
    for node in fileh.list_nodes(fileh.root.label):    
        labelDict[node._v_name] = node.read()
    fileh.close()
    return labelDict
    
############################ DISTANCE MEASURE ############################
def calculateEuclidDistance(vectorFeature1, vectorFeature2):
    return np.linalg.norm(vectorFeature1-vectorFeature2)

def calculateCosineDistance(vectorFeature1, vectorFeature2):
    return cosine(vectorFeature1,vectorFeature2)

def getDistanceVal(vectorFeature1, vectorFeature2):
#    return calculateEuclidDistance(vectorFeature1, vectorFeature2)
    return calculateCosineDistance(vectorFeature1, vectorFeature2)


############################ BUILD MATRIX ############################
#def getWeight(xi, xj,t=2):
#    exponent = -math.fabs(xi-xj)/t
#    return math.exp(exponent)

def getWeight(isclose):
    if isclose == True:
        return 3
    else:
        return 1
    
def getLastIndexLabel(labelDict):
    lastIndex = -1
    for label, listIndexSeq in labelDict.iteritems():
        maxIndex = max(listIndexSeq)
        if maxIndex > lastIndex:
            lastIndex = maxIndex   
    return lastIndex

def findKNN(seqVec, indexOfCurSeq, seqRowMat, kNumOfNeighbors):
    sizeOfseqRowMat = len(seqRowMat)
    distanceVec = np.zeros(sizeOfseqRowMat)
    distanceVec = np.array([getDistanceVal(seqVec, seq) for seq in seqRowMat]) 
    sortedIndexList = np.argsort(distanceVec)
    sortedIndexList = np.delete(sortedIndexList, indexOfCurSeq)
    return sortedIndexList[:kNumOfNeighbors]
    

def buildWeightMat(seqRowMat, labelDict, kNumOfNeighbors):
    numOfSeq = len(seqRowMat)
    weightMat = np.zeros((numOfSeq, numOfSeq))
    
    listIndexSeqs = labelDict.values()
    numOfLabel = len(listIndexSeqs)
    ######## cho cac labeled data ########
    # danh weight cho seq cung label
    for listIndexSeq in listIndexSeqs:
        sizeListOfCurLabel = len(listIndexSeq)
        for index1 in range(0,sizeListOfCurLabel):
            for index2 in range(index1+1,sizeListOfCurLabel):
                weightMat[index1][index2] = weightMat[index2][index1] = getWeight(isclose=True)
                
    # danh weight cho seq khac label
    for indexListIndexSeq1 in range(0,numOfLabel):
        for indexListIndexSeq2 in range(indexListIndexSeq1+1,numOfLabel):
            listIndexSeq1 = listIndexSeqs[indexListIndexSeq1] #list cac index co label1
            listIndexSeq2 = listIndexSeqs[indexListIndexSeq2] #list cac index co label2
            for index1 in listIndexSeq1:
                for index2 in listIndexSeq2:
                    weightMat[index1][index2] = weightMat[index2][index1] = getWeight(isclose=False)
 
    ######## cho cac unlabeled data ########
    startIndexUnlabeledData = getLastIndexLabel(labelDict) + 1
    for indexUnlabeledData in range(startIndexUnlabeledData, numOfSeq):
        listKNNIndexes = findKNN(seqRowMat[indexUnlabeledData], indexUnlabeledData, seqRowMat, kNumOfNeighbors)
        # danh weight cho cac close data point
        for neighborIndex in listKNNIndexes:
            weightMat[indexUnlabeledData][neighborIndex] = weightMat[neighborIndex][indexUnlabeledData] = getWeight(isclose = True)
    
    return weightMat



def buildDiagleMat(weightMat):
    shapeOfWeightMat = np.shape(weightMat)
    numOfRow = shapeOfWeightMat[0]
    numOfCol = shapeOfWeightMat[1] 
    vectorOnes = np.ones((numOfCol,1))
    #weightMat.vectorOnes --> shape(numOfRow,1) -->reshape ve (1,numOfRow) de dung ham chuyen mtran duong cheo
    return np.diag(np.matmul(weightMat,vectorOnes).reshape(numOfRow))  
    

def buildLaplacianMat(weightMat, diagleMat):
    return diagleMat - weightMat

############################ COMPUTE SCORE ############################
#2 ham nomarlizeFeature giong nhau chi khac o cach tinh mean
#def normalizeFeature1(vectorFeature):
#    size = len(vectorFeature)
#    meanFeature = np.mean(vectorFeature)
#    return vectorFeature - np.array([[meanFeature]]*size)    

def normalizeFeature2(vectorFeature, diagleMat):
    size = len(vectorFeature)
    vectorOnes = np.ones((size,1))
    meanFeature = ((np.transpose(vectorFeature).dot(diagleMat).dot(vectorOnes))
                        / (np.transpose(vectorOnes).dot(diagleMat).dot(vectorOnes)))
    return vectorFeature - meanFeature*vectorOnes
    # or return vectorFeature - np.array([[meanFeature]]*size)
    

def computeLaplacianScore(vectorFeature,laplaMat, diagleMat):
    normFeatureVec = normalizeFeature2(vectorFeature,diagleMat)
    return np.asscalar((np.transpose(normFeatureVec).dot(laplaMat).dot(normFeatureVec))
                / (np.transpose(normFeatureVec).dot(diagleMat).dot(normFeatureVec))) #np.asscalar convert array 1 element to scarlar
    

################################### OTHER ###################################
def deleteByValNumpyArr(array, value):
    index = np.argwhere(array==value)
    return np.delete(array, index)
    
################################### MAIN ###################################
    
def getKBestFeatureIndex(listLaplacianScore, k=10):
    sortedIndexLaplaScoreList = np.argsort(listLaplacianScore)
    return sortedIndexLaplaScoreList[:k]

if __name__ == '__main__':
    from datetime import datetime
    startTime = datetime.now()
    kNumOfNeighbors = 100
    seqRowMat = readSeqRowMatFromFile()
    labelDict = readLabelDictFromFile()
    weightMat = buildWeightMat(seqRowMat, labelDict, kNumOfNeighbors)
    del seqRowMat
    diagleMat = buildDiagleMat(weightMat)
    laplaMat = buildLaplacianMat(weightMat, diagleMat)  
    
    
    featureRowMat = readFeatureRowMatFromFile()
    listLaplacianScore = [computeLaplacianScore(np.transpose([featureVec]),laplaMat, diagleMat) for featureVec in featureRowMat]
    del weightMat
    del diagleMat
    del laplaMat
    
    #get k best features
    kBestFeatureIndex = getKBestFeatureIndex(listLaplacianScore, kNumOfNeighbors)
    
    #save new seqRowMatrix to file after feature selection
    newFeatureRowMat = np.array([featureRowMat[index] for index in kBestFeatureIndex])
    
    #save summarization file after feature selection
    fileh = tables.open_file('..//outputfile//afterFeatureSelection.h5','w')
    root = fileh.root
    labelGroup = fileh.create_group(root, "label")
    
    for key in labelDict:
        fileh.create_array(labelGroup, key, labelDict[key], key)
    
    fileh.create_array(root, 'seqRowMatrix', newFeatureRowMat.T,"seqRowMatrix")
    featureList = readFeatureListFromFile()
    fileh.create_array(root, 'feature', np.array([featureList[index] for index in kBestFeatureIndex]),"feature")
    fileh.flush()
    fileh.close()
    
    print datetime.now() - startTime



    
    

    
    

    


