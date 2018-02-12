# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from datetime import datetime
from scipy import sparse

from Bio import SeqIO
startTime = datetime.now()

#-----------------------------GET SEQUENCES FROM FASTA FILE-----------------------
seqList = []

inputfileList = ('..//inputfile//HCV_nonlabel.fasta','..//inputfile//HCV_label.fasta', '..//inputfile//sequence.txt')
for fileName in inputfileList:
    inFile = open(fileName,'r')
    
    for record in SeqIO.parse(inFile,'fasta'):
        seqList.append( ''.join(char for char in str(record.seq) if char.isalpha()))
    inFile.close()
        

print datetime.now() - startTime #khoảng 13-14 gi


#------------------------------ENCODING-------------------------------------
sizeOfSubstr = (1,2)
numOfSeq = len(seqList)
listSubString = [] #list of substrings or list of feature
#encodingMat= np.zeros((0,numOfSeq), dtype = 'uint8') #dòng là feature(substr), cột là seq
encodingMat = sparse.lil_matrix((0,numOfSeq), dtype = 'uint8')

for sizeOfSub in sizeOfSubstr: #số lượng kí tự của substr
    curNumOfFeature = len(listSubString)
    startIndexOfCurSize = curNumOfFeature
    
    for indexOfCurSeq in xrange(0,numOfSeq): #duyệt từng sequence
        seq = seqList[indexOfCurSeq]
        sizeOfSeq = len(seq)

        for index in xrange(0,sizeOfSeq-sizeOfSub+1): # duyệt substr
            curSubStr = seq[index: index+sizeOfSub]
            try:
                #chỉ tìm trong các sub string có cùng số lượng kí tự nên có startIndexOfCurSize
                foundIndexOfFeature = listSubString.index(curSubStr, startIndexOfCurSize)
                #nếu có cho thêm sửa value trong matrix
                encodingMat[foundIndexOfFeature,indexOfCurSeq] += 1
            except ValueError:
                #nếu không tìm được
                #thêm feature vào list
                listSubString.append(curSubStr)
                #tạo thêm dòng feature trong matrix
                #tạo row
                newRowFeature = np.zeros((numOfSeq), dtype='uint8')
                #sửa gtri feature trong dòng của seq tạo ra feature mới này
                newRowFeature[indexOfCurSeq] += 1 
                #add row vào trong matrix
                encodingMat = sparse.vstack([encodingMat, newRowFeature], format = 'lil', dtype = 'uint8')
                
#--------------------------------
for i in xrange(100):
    print (encodingMat.getrowview(i))
                
##save mat by numpy
#np.save('output2',encodingMat)
#readMat1 = np.load('output2.npy')
#
#
##save sparse Mat to .npz file by scipy.sparse lib
#sparseMat = csc_matrix(encodingMat)
#save_npz('output2', sparseMat)
#
#
#readMat = load_npz('output2.npz')
#readMat = readMat.todense()

##-------------------------------------
##save sparse Mat to .npz file by scipy.sparse lib
#sparse.save_npz('output3', encodingMat)
#
#
#readMat = sparse.load_npz('output3.npz')
#readMat = readMat.todense()

print datetime.now() - startTime
