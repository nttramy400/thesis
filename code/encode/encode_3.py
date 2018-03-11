#######################################################
from datetime import datetime

from Bio import SeqIO
import numpy as np
import tables

startTime = datetime.now()

#-----------------------------GET SEQUENCES FROM FASTA FILE-----------------------
#seqList = []
#
#inputfileList = ('..//inputfile//HCV_nonlabel.fasta','..//inputfile//HCV_label.fasta', '..//inputfile//sequence.txt')
#for fileName in inputfileList:
#    inFile = open(fileName,'r')
#    
#    for record in SeqIO.parse(inFile,'fasta'):
#        seqList.append( ''.join(char for char in str(record.seq) if char.isalpha()))
#    inFile.close()
#        
#
#print datetime.now() - startTime #khoang 13-14s

#----------------------------GET SET SUBSTRING---------------------------------------
sizeOfSubstr = range(2,20)

setSubString = set()

for sizeOfSub in sizeOfSubstr: 
    for seq in seqList: 
        sizeOfSeq = len(seq)
        for index in xrange(0,sizeOfSeq-sizeOfSub+1): 
            setSubString.add(seq[index: index+sizeOfSub])

print datetime.now() - startTime #khoang 2 phut

subStrFile = open('substring.txt', 'w')

for substr in setSubString:
    subStrFile.write("%s\n" % substr)
    
subStrFile.flush()
subStrFile.close()

#-------------------------BUILD MATRIX AND SAVE TO HDF5 FILES--------------------------

sizeOfSeqList =  len(seqList)
# Mo 1 empty HDF5 file moi
fileh = tables.open_file("relationalMat.h5", mode="w")

# Lay root cua group
root = fileh.root
atom = tables.Int16Atom()
# Use ``a`` as the object type for the enlargeable array.
matrix = fileh.create_earray(fileh.root, 'matrix', atom, (0,sizeOfSeqList),"FeatureRows")
count = 0
listSubStr = list(setSubString)
for feature in listSubStr:
    a = np.zeros((sizeOfSeqList,), np.uint16)
    
    #dem so lan xuat hien
    for indexOfSeq in xrange(sizeOfSeqList):
        a[indexOfSeq] = seqList[indexOfSeq].count(feature)
    
    # luu array vao HDF5 file
    matrix.append([a])
    count+=1
    print (count)
    print datetime.now() - startTime

fileh.flush()
fileh.close()

print datetime.now() - startTime #khoang 6 tieng  ch6 7309 seqs lấy size feature = 2-->19 (3373908 features)