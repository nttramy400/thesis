# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from datetime import datetime

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
        

print datetime.now() - startTime #khoang 13-14s

#----------------------------GET SET SUBSTRING---------------------------------------
sizeOfSubstr = [2,3,4]

setSubString = set()

for sizeOfSub in sizeOfSubstr: 
    for seq in seqList: 
        sizeOfSeq = len(seq)
        for index in xrange(0,sizeOfSeq-sizeOfSub+1): 
            setSubString.add(seq[index: index+sizeOfSub])

print datetime.now() - startTime #khoang 2 phut


##----------------------------GET LIST SUBSTRING---------------------------------------
#sizeOfSubstr = [1,2,3]
#
#listSubString = []
#
#for sizeOfSub in sizeOfSubstr: 
#    startIndexOfCurSize = len(listSubString)
#    for seq in seqList: 
#        sizeOfSeq = len(seq)
#        for index in xrange(0,sizeOfSeq-sizeOfSub+1): 
#            curSubStr = seq[index: index+sizeOfSub]
#            try:
#                foundIndex = listSubString.index(curSubStr, startIndexOfCurSize)
#            except ValueError:
#                listSubString.append(curSubStr)
#
#
#print datetime.now() - startTime #khoang 42 phut




