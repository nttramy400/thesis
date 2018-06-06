import numpy as np
import tables
from datetime import datetime
startTime = datetime.now()
version = "10"
#-----------------------------GET SEQUENCES FROM INFO FILE-----------------------
infoFile = tables.open_file("..//outputfile//"+version+"//info.h5",mode = 'r')
seqList = infoFile.root.sequences.read()
infoFile.close()

#----------------------------GET SET SUBSTRING---------------------------------------
fileh = tables.open_file("..//outputfile//"+version+"//encodingFile.h5", mode="w")
# Lay root cua file
root = fileh.root

#seqList = ['abc', 'abca']
sizeOfSubstr = range(2,41)

#setSubString = set()
dictSubString = {}

for sizeOfSub in sizeOfSubstr: 
    for seq in seqList: 
        sizeOfSeq = len(seq)
        for index in xrange(0,sizeOfSeq-sizeOfSub+1): 
            subSeq = seq[index: index+sizeOfSub]
#            setSubString.add(subSeq)
            if subSeq in dictSubString: #count so lan xuat hien sub subStr
                dictSubString[subSeq] += 1
            else:
                dictSubString[subSeq] = 1

#listSubStr = list(setSubString)
#listSubStr = dictSubString.keys()
#testFreq = filter(lambda key: (len(key)>10 and dictSubString[key] >10),dictSubString)  #so luong subSeq co len>10 va co freq>10
listSubStr = filter(lambda key: dictSubString[key] > 0.5*len(seqList), dictSubString) #lay cac subSeq co tan so xuat hien > 10


print datetime.now() - startTime #khoang 2 phut

"""
tao tuple gom index cac array giong array dang xet
co chua luon index cua array dang xet
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

#-------------------------BUILD MATRIX AND SAVE TO HDF5 FILES--------------------------
sizeOfSeqList =  len(seqList)

atom = tables.Int16Atom()
# Use ``a`` as the object type for the enlargeable array.
featureRowMatrix = fileh.create_earray(root, 'featureRowMatrix', atom, (0,sizeOfSeqList),"featureRowMatrix")
seqRowMatrix = fileh.create_earray(root, 'seqRowMatrix', atom, (sizeOfSeqList,0),"seqRowMatrix")
count = 0



for curFeatureIndex in range(len(listSubStr)):
    count+=1
    print (count)
    curFeature = listSubStr[curFeatureIndex]
    arr = np.zeros((sizeOfSeqList,), np.uint16)
    
    #dem so lan xuat hien
    for indexOfSeq in xrange(sizeOfSeqList):
        arr[indexOfSeq] = seqList[indexOfSeq].count(curFeature)
        
    # luu array vao HDF5 file
    featureRowMatrix.append([arr])
    seqRowMatrix.append(arr.reshape(sizeOfSeqList,1))
    
    print datetime.now() - startTime



def isNumIn2DList(num, list2DNum):
    for arr in list2DNum:
        for curNum in arr:
            if curNum == num:
                return True
    return False


#group cac feature co vector feature giong nhau
groupDuplicateFeatureRow = []
for curFeatureIndex in range(len(listSubStr)):
    print (curFeatureIndex)
    # da nam trong 1 group duplicate nao do khong can xet tiep
    if isNumIn2DList(curFeatureIndex,groupDuplicateFeatureRow):
        pass
    else:
        #check xem co bi duplicate k
        groupDupIndex = getGroupIndexOfSameArrInNdarray(featureRowMatrix,curFeatureIndex, listSubStr)
        if groupDupIndex != None: # bi duplicate
            groupDuplicateFeatureRow.append(groupDupIndex)

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
            
#
#lap list index cua cac feature duplicate can loai bo
listDuplicateFeatureRowIndex =[]  
for group in groupDuplicateFeatureRow:
    #lay feature co do dai lon nhat, bo cac feature la con cua no khi cac vector giong nhau
    selectedFeatureIndex = getIndexOfLongestStrInListIndex(group,listSubStr)
    group.remove(selectedFeatureIndex)
    listDuplicateFeatureRowIndex = listDuplicateFeatureRowIndex + group

#remove feature duplicate 
featureRowMatrix= np.delete(featureRowMatrix,listDuplicateFeatureRowIndex,axis = 0)
listSubStr= np.delete(np.array(listSubStr),listDuplicateFeatureRowIndex,axis = 0)

#save again
fileh.root.featureRowMatrix._f_remove()
fileh.root.seqRowMatrix._f_remove()
featureRowMatrix = fileh.create_array(root, 'featureRowMatrix', featureRowMatrix,"featureRowMatrix")
seqRowMatrix = fileh.create_array(root, 'seqRowMatrix', np.transpose(featureRowMatrix),"seqRowMatrix")

#save features to file
fileh.create_array(root, 'feature', listSubStr, 'feature')
fileh.create_array(root, 'nOfNgram', sizeOfSubstr, 'nOfNgram')
fileh.flush()
fileh.close()

print datetime.now() - startTime #khoang 6h10'  cho 7309 seqs lấy size feature = 2-->19 (3373908 features)