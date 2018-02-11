# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 04:39:31 2018

@author: BeoU
"""
from datetime import datetime  
from Bio import SeqIO
startTime = datetime.now()

#seqList = []
#list2 = []
#list3= []
#
#inputfileList = ('..//inputfile//HCV.fasta', '..//inputfile//sequence.txt')
#for fileName in inputfileList:
#    inFile = open(fileName,'r')
#    
#    for record in SeqIO.parse(inFile,'fasta'):
#        seqList.append( ''.join(char for char in str(record.seq) if char.isalpha()))
#        
#
#print datetime.now() - startTime

    
#from datetime import datetime    
#from Bio import SeqIO
#startTime = datetime.now()
#list1 = []
#list2 = []
#with open('..//inputfile//sequence.fasta', "rU") as handle:
#    for record in SeqIO.parse(handle, "fasta"):
#        list2.append(record.id)
#        list1.append( ''.join(e for e in record.seq._data if e.isalpha()))
#
#    
#print datetime.now() - startTime


#------------------------------------------------

none_label_seqs =[]
inFile = open('..//inputfile//HCV_nonlabel.fasta','r')
    
for record in SeqIO.parse(inFile,'fasta'):
    none_label_seqs.append( ''.join(char for char in str(record.seq) if char.isalpha()))


label_seqs =[]
inFile = open('..//inputfile//HCV_label.fasta','r')
    
for record in SeqIO.parse(inFile,'fasta'):
    label_seqs.append( ''.join(char for char in str(record.seq) if char.isalpha()))



