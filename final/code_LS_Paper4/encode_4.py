import numpy as np
import tables
import logging
import h5filereader
logging.basicConfig(level=logging.DEBUG)


def getListFeature(seqList, thresholdFreq=0, sizeOfSubstr = range(2,41)):
    logging.info("Get all list substring from raw ISDR sequences")
    dictSubStrFrequency ={}
    for sizeOfSub in sizeOfSubstr: 
        for seq in seqList: 
            sizeOfSeq = len(seq)
            for index in xrange(0,sizeOfSeq-sizeOfSub+1): 
                subSeq = seq[index: index+sizeOfSub]
                if subSeq in dictSubStrFrequency: #count so lan xuat hien sub subStr
                    dictSubStrFrequency[subSeq] += 1
                else:
                    dictSubStrFrequency[subSeq] = 1

    listFeature = filter(lambda key: dictSubStrFrequency[key] > thresholdFreq, dictSubStrFrequency) #lay cac subSeq co tan so xuat hien > thresholdFreq
    return listFeature


#build relational matrix 
def buildRelationalMatrixToH5File(seqList, listFeature, filename = "encodingFile.h5"):
    logging.info("Build relational matrix")
    fileh = tables.open_file(filename, mode="w")
    # Lay root cua file
    root = fileh.root
    sizeOfSeqList =  len(seqList)

    atom = tables.Int16Atom()

    featureRowMatrix = fileh.create_earray(root, 'featureRowMatrix', atom, (0,sizeOfSeqList),"featureRowMatrix")
    seqRowMatrix = fileh.create_earray(root, 'seqRowMatrix', atom, (sizeOfSeqList,0),"seqRowMatrix")

    for curFeatureIndex in range(len(listFeature)):
#        print (curFeatureIndex)
        curFeature = listFeature[curFeatureIndex]
        arr = np.zeros((sizeOfSeqList,), np.uint16)
        
        #dem so lan xuat hien
        for indexOfSeq in xrange(sizeOfSeqList):
            arr[indexOfSeq] = seqList[indexOfSeq].count(curFeature)
            
        # luu array vao HDF5 file
        featureRowMatrix.append([arr])
        seqRowMatrix.append(arr.reshape(sizeOfSeqList,1))
    fileh.flush()
    fileh.close()

"""
tao tuple gom index cac array giong array dang xet
co chua luon index cua array dang xet
muc dich de remove duplicate feature Vecto
ham bo tro cho groupDuplicateFeatureRow
"""
def getGroupIndexOfSameArrInNdarray(matrix, indexArr, listSubStr):
    listRet = []
    numOfRowMat = len(matrix)
    for curIndex in range(numOfRowMat):
        if np.array_equal(matrix[curIndex],matrix[indexArr]):
            if (listSubStr[curIndex] in listSubStr[indexArr]) or (listSubStr[indexArr] in listSubStr[curIndex]):
                listRet.append(curIndex)
    
    #chi chua 1 index duy nhat la chinh no ==> khong co arr nao giong no
    if (len(listRet) == 1):
        return None
    else:
        return listRet
    
#remove duplicate feature vector
def isNumIn2DList(num, list2DNum):
    for arr in list2DNum:
        for curNum in arr:
            if curNum == num:
                return True
    return False

#group cac feature co vector feature giong nhau
def getGroupDuplicateFeatureRow(listFeature, featureRowMatrix):
    groupDuplicateFeatureRow = []
    for curFeatureIndex in range(len(listFeature)):
#        print (curFeatureIndex)
        # da nam trong 1 group duplicate nao do khong can xet tiep
        if isNumIn2DList(curFeatureIndex,groupDuplicateFeatureRow):
            pass
        else:
            #check xem co bi duplicate k
            groupDupIndex = getGroupIndexOfSameArrInNdarray(featureRowMatrix,curFeatureIndex, listFeature)
            if groupDupIndex != None: # bi duplicate
                groupDuplicateFeatureRow.append(groupDupIndex)
    return groupDuplicateFeatureRow

"""
tim index cua substring co length lon nhat trong cac substr co index trong listIndexSubStr
"""
def getIndexOfLongestStrInListIndex(listIndexSubStr, listAllSubStr):
    maxLength = -1
    retIndex = -1
    for curIndexSubStr in listIndexSubStr:
        curLength = len(listAllSubStr[curIndexSubStr])
        if  curLength > maxLength:
            maxLength = curLength
            retIndex = curIndexSubStr
    return retIndex

#lap list index cua cac feature duplicate can loai bo
def getListDuplicateFeatureRowIndex(listFeature,groupDuplicateFeatureRow):
    logging.info("Find all indexes of duplicate feature vector")
    listDuplicateFeatureRowIndex =[]  
    for group in groupDuplicateFeatureRow:
        #lay feature co do dai lon nhat, bo cac feature la con cua no khi cac vector giong nhau
        selectedFeatureIndex = getIndexOfLongestStrInListIndex(group,listFeature)
        group.remove(selectedFeatureIndex)
        listDuplicateFeatureRowIndex = listDuplicateFeatureRowIndex + group
    return listDuplicateFeatureRowIndex

#remove feature vecto duplicate 
def removeDuplicateFeatureVec(featureRowMatrix, listFeature, listDuplicateFeatureRowIndex):
    logging.info("Remove feature vectors which are duplicate")
    newFeatureRowMatrix = np.delete(featureRowMatrix,listDuplicateFeatureRowIndex,axis = 0)
    newListFeature= np.delete(np.array(listFeature),listDuplicateFeatureRowIndex,axis = 0)
    return newFeatureRowMatrix, newListFeature

#luu relational table moi thay the cai da co
def replaceRelationalMatFromH5File(newFeatureRowMatrix, filename = "encodingFile.h5"):
    logging.info("Overwrite existed relational matrix in h5 file")
    fileh = tables.open_file(filename, mode="a")
    root = fileh.root
    root.featureRowMatrix._f_remove()
    root.seqRowMatrix._f_remove()
    fileh.create_array(root, 'featureRowMatrix', newFeatureRowMatrix,"featureRowMatrix")
    fileh.create_array(root, 'seqRowMatrix', np.transpose(newFeatureRowMatrix),"seqRowMatrix")
    fileh.flush()
    fileh.close()
    
def saveInfoFeatureToH5File(listFeature,sizeOfSubstr=range(2,41), filename = "encodingFile.h5"):
    logging.info("Save infomation of features to h5 file")
    fileh = tables.open_file(filename, mode="a")
    root = fileh.root
    fileh.create_array(root, 'feature', listFeature, 'feature')
    fileh.create_array(root, 'nOfNgram', sizeOfSubstr, 'nOfNgram')
    fileh.flush()
    fileh.close()
    
def encode():
    logging.info("Encoding...")
    #get feature
    sizeOfSubstr = range(2,41)
    seqList = h5filereader.getSeqListFromH5File()
    listFeature = getListFeature(seqList,thresholdFreq=0.25*len(seqList), sizeOfSubstr= sizeOfSubstr)
    
    #build relational Matrix & save to H5 file
    buildRelationalMatrixToH5File(seqList,listFeature)
    
    #remove duplicate feature Row        
    fileh = tables.open_file("encodingFile.h5", mode="r")
    featureRowMatrix =fileh.root.featureRowMatrix
    groupDuplicateFeatureRow = getGroupDuplicateFeatureRow(listFeature, featureRowMatrix)
    
    
    listDuplicateFeatureRowIndex = getListDuplicateFeatureRowIndex(listFeature,groupDuplicateFeatureRow)
    newFeatureRowMatrix, newListFeature = removeDuplicateFeatureVec(featureRowMatrix,listFeature,listDuplicateFeatureRowIndex)
    fileh.close()
    
    #save info again
    replaceRelationalMatFromH5File(newFeatureRowMatrix)
    saveInfoFeatureToH5File(newListFeature,sizeOfSubstr)
    
if __name__ == "__main__":
    encode()
    