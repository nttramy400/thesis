import numpy as np
import tables

startTime = datetime.now()

#-----------------------------GET SEQUENCES FROM INFO FILE-----------------------
infoFile = tables.open_file('..//outputfile//info.h5',mode = 'r')
seqList = infoFile.root.sequences.read()
infoFile.close()

#----------------------------GET SET SUBSTRING---------------------------------------
fileh = tables.open_file("..//outputfile//encodingFile.h5", mode="w")
# Lay root cua file
root = fileh.root

#seqList = ['abc', 'abca']
sizeOfSubstr = range(2,20)

setSubString = set()

for sizeOfSub in sizeOfSubstr: 
    for seq in seqList: 
        sizeOfSeq = len(seq)
        for index in xrange(0,sizeOfSeq-sizeOfSub+1): 
            setSubString.add(seq[index: index+sizeOfSub])

listSubStr = list(setSubString)
#save features to file
fileh.create_array(root, 'feature', listSubStr, 'feature')
fileh.create_array(root, 'nOfNgram', sizeOfSubstr, 'nOfNgram')

print datetime.now() - startTime #khoang 2 phut

#-------------------------BUILD MATRIX AND SAVE TO HDF5 FILES--------------------------

sizeOfSeqList =  len(seqList)

atom = tables.Int16Atom()
# Use ``a`` as the object type for the enlargeable array.
featureRowMatrix = fileh.create_earray(root, 'featureRowMatrix', atom, (0,sizeOfSeqList),"featureRowMatrix")
seqRowMatrix = fileh.create_earray(root, 'seqRowMatrix', atom, (sizeOfSeqList,0),"seqRowMatrix")
count = 0

for feature in listSubStr:
    a = np.zeros((sizeOfSeqList,), np.uint16)
    
    #dem so lan xuat hien
    for indexOfSeq in xrange(sizeOfSeqList):
        a[indexOfSeq] = seqList[indexOfSeq].count(feature)
    
    # luu array vao HDF5 file
    featureRowMatrix.append([a])
    seqRowMatrix.append(a.reshape(sizeOfSeqList,1))
    count+=1
    print (count)
    print datetime.now() - startTime

fileh.flush()
fileh.close()

print datetime.now() - startTime #khoang 6h10'  cho 7309 seqs lấy size feature = 2-->19 (3373908 features)