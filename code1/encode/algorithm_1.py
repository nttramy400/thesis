import numpy as np
import tables
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
version = "9"
testVer = "5"
############################ READ INFO FROM FILE ############################
def readFeatureRowMatFromFile(filename="..//outputfile//"+version+"//encodingFile.h5"):
    fileh = tables.open_file(filename, mode="r")        
    featureRowMatrix = fileh.root.featureRowMatrix.read()
    fileh.close()
    return featureRowMatrix
    
def readSeqRowMatFromFile(filename="..//outputfile//"+version+"//encodingFile.h5"):
    fileh = tables.open_file(filename, mode="r")        
    seqRowMatrix = fileh.root.seqRowMatrix.read()
    fileh.close()
    return seqRowMatrix

def readFeatureListFromFile(filename="..//outputfile//"+version+"//encodingFile.h5"):
    fileh = tables.open_file(filename, mode="r")        
    featureList = fileh.root.feature.read()
    fileh.close()
    return featureList

def readLabelVecFromFile(filename="..//outputfile//"+version+"//info.h5"):
    fileh = tables.open_file(filename, mode="r")
    labelVec = fileh.root.labelVec.read()
    fileh.close()
    return labelVec
    
############################ DISTANCE MEASURE ############################
def calculateEuclidDistance(vectorFeature1, vectorFeature2):
    return np.linalg.norm(vectorFeature1-vectorFeature2)

def calculateCosineDistance(vectorFeature1, vectorFeature2):
    return cosine(vectorFeature1,vectorFeature2)

def getDistanceVal(vectorFeature1, vectorFeature2):
    return calculateEuclidDistance(vectorFeature1, vectorFeature2)
#    return calculateCosineDistance(vectorFeature1, vectorFeature2)


############################ BUILD MATRIX ############################
#def getWeight(xi, xj,t=2):
#    exponent = -math.fabs(xi-xj)/t
#    return math.exp(exponent)

def getWeight(isclose=None, seqVec1=None, seqVec2=None, isLabeled=False):
    if isclose==True and isLabeled==True:
        return 1
    if isclose == True:
#        return np.exp(-calculateCosineDistance(seqVec1,seqVec2))
#        return np.exp(-pow(calculateCosineDistance(seqVec1,seqVec2),2))
#        return 2
        return 1-calculateCosineDistance(seqVec1,seqVec2)
    else:
        return 0

#def findKNN(seqVec, indexOfCurSeq, seqRowMat, kNumOfNeighbors):
#    sizeOfseqRowMat = len(seqRowMat)
#    distanceVec = np.zeros(sizeOfseqRowMat)
#    distanceVec = np.array([getDistanceVal(seqVec, seq) for seq in seqRowMat])
#    
#    sortedIndexList = np.argsort(distanceVec)
#    sortedIndexList = np.delete(sortedIndexList, indexOfCurSeq)
#    return sortedIndexList[:kNumOfNeighbors]
    

def buildWeightMat(seqRowMat, labelVec, kNumOfNeighbors):
    numOfSeq = len(seqRowMat)
    weightMat = np.zeros((numOfSeq, numOfSeq))
    
    numOfLabelSeq = len(labelVec)

    ######## cho cac labeled data ########
    # danh weight cho seq cung label    
    for index1 in range(numOfLabelSeq):
        for index2 in range(index1, numOfLabelSeq):
            if labelVec[index1] == labelVec[index2]: 
                weightMat[index1][index2] = weightMat[index2][index1] \
                = getWeight(isclose=True, isLabeled=True)
                    
    ######## danh weight cho cac unlabeled data ########
    #tim KNN cho tat ca data point
    nbrs = NearestNeighbors(n_neighbors=kNumOfNeighbors, metric = 'cosine').fit(seqRowMat)
    ndistances, indices = nbrs.kneighbors(seqRowMat)

    startIndexUnlabeledData = numOfLabelSeq
    for indexUnlabeledData in range(startIndexUnlabeledData, numOfSeq):
#        listKNNIndexes = findKNN(seqRowMat[indexUnlabeledData], indexUnlabeledData, seqRowMat, kNumOfNeighbors)
        listKNNIndexes = indices[indexUnlabeledData]
        # danh weight cho cac close data point
        for neighborIndex in listKNNIndexes:
            if neighborIndex != indexUnlabeledData:
                weightMat[indexUnlabeledData][neighborIndex] \
                = weightMat[neighborIndex][indexUnlabeledData] \
                = getWeight(isclose=True,seqVec1=seqRowMat[neighborIndex],seqVec2=seqRowMat[indexUnlabeledData])
    
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

def saveSortedLaplaFeatureIndexes(listLaplacianScore, filename = "..//outputfile//"+version+"//sortedLaplaFeatureIndexes"):
    sortedLaplaFeatureIndexes = np.argsort(listLaplacianScore)
    np.save(filename, sortedLaplaFeatureIndexes)

def loadSortedLaplaFeatureIndexes(filename = "..//outputfile//"+version+"//sortedLaplaFeatureIndexes.npy"):
    return np.load(filename)    

if __name__ == '__main__':
    from datetime import datetime
    startTime = datetime.now()
#    kNumOfNeighbors = 115 
#    
#    seqRowMat = readSeqRowMatFromFile()
#    labelDict = readLabelDictFromFile()
#    weightMat = buildWeightMat(seqRowMat, labelDict, kNumOfNeighbors)
#    print datetime.now() - startTime
#    del seqRowMat
#    diagleMat = buildDiagleMat(weightMat)
#    laplaMat = buildLaplacianMat(weightMat, diagleMat)  
#    
#    
#    featureRowMat = readFeatureRowMatFromFile()
#    listLaplacianScore = [computeLaplacianScore(np.transpose([featureVec]),laplaMat, diagleMat) for featureVec in featureRowMat]
#    #save to file
#    saveSortedLaplaFeatureIndexes(listLaplacianScore)
#    del weightMat
#    del diagleMat
#    del laplaMat
    ##-----------------------------------------------------------------------------------------------
#    kFeatures = 175
#    #get k best features
#    kBestFeatureIndex = getKBestFeatureIndex(listLaplacianScore, kFeatures)
#    
#    #save new seqRowMatrix to file after feature selection
#    newFeatureRowMat = np.array([featureRowMat[index] for index in kBestFeatureIndex])
#    
#    #save summarization file after feature selection
#    fileh = tables.open_file('..//outputfile//afterFeatureSelection.h5','w')
#    root = fileh.root
#    labelGroup = fileh.create_group(root, "label")
#    
#    for key in labelDict:
#        fileh.create_array(labelGroup, key, labelDict[key], key)
#    
#    fileh.create_array(root, 'seqRowMatrix', newFeatureRowMat.T,"seqRowMatrix")
#    featureList = readFeatureListFromFile()
#    fileh.create_array(root, 'feature', np.array([featureList[index] for index in kBestFeatureIndex]),"feature")
#    fileh.flush()
#    fileh.close()
    
    ##----------------------------------------------------------------------------------------
    
    
    for kNumOfNeighbors in range(1,200):
    
        seqRowMat = readSeqRowMatFromFile()
        labelVec = readLabelVecFromFile()
        weightMat = buildWeightMat(seqRowMat, labelVec, kNumOfNeighbors)
#        print datetime.now() - startTime
        del seqRowMat
        diagleMat = buildDiagleMat(weightMat)
        laplaMat = buildLaplacianMat(weightMat, diagleMat)  
        
        
        featureRowMat = readFeatureRowMatFromFile()
        listLaplacianScore = [computeLaplacianScore(np.transpose([featureVec]),laplaMat, diagleMat) for featureVec in featureRowMat]
        #save to file
        saveSortedLaplaFeatureIndexes(listLaplacianScore, filename = ("..//outputfile//"+version+ "//"+testVer+"//knnGraph_"+str(kNumOfNeighbors)))
        del weightMat
        del diagleMat
        del laplaMat
        print datetime.now() - startTime
    
    print datetime.now() - startTime



    
    

    
    

    




