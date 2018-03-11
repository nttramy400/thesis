from datetime import datetime  
from Bio import SeqIO
startTime = datetime.now()
#---------------------------- GET SEQUENCES FROM FILES -----------------------
#################### KHONG NHAN
none_label_seqs = set() #loai cac seq giống nhau


inFile = open('..//inputfile//HCV_nonlabel.txt','r')

for record in SeqIO.parse(inFile,'fasta'):
    if 'ns5a' in record.description.lower():
        none_label_seqs.add(str(record.seq))

none_label_seqs = list(none_label_seqs)
inFile.close()

#################### CO NHAN
#********Tu file fasta LANL
#co 2 class la sustained_response, non-response
labelClass = {"sustained-response":["sustained_response", "SVR"],
              "non-response":["non-response", "relapse", "Null"]}
seqClass = {"sustained-response":[], "non-response":[]} #luu index cua sequence tung loai

def getLabelRecord(record):
    return str(record.description).split('.')[1] # 1 la vi tri cua nhan trong phan description cua file fasta

def getClass(nameLabel):
    for key in labelClass:
        if nameLabel in labelClass[key]:
            return key
    return None

def addLabelSeq(record):
    global seqClass
    global label_seqs
    labelSeq = getLabelRecord(record)
    seqClass[getClass(labelSeq)].append(len(label_seqs)) # them index cua chuoi vao dung class
    label_seqs.append(str(record.seq))
  

label_seqs = []
#luu index cua seq co nhan do


inFile = open('..//inputfile//HCV_label.fasta','r')

for record in SeqIO.parse(inFile,'fasta'):
    if str(record.seq) in label_seqs:       
        indexSameSeq = label_seqs.index(str(record.seq))
        labelCurRecordSeq = getLabelRecord(record)
        
        # 2 seq giong nhau co cung nhan thi bo, khac nhan thi luu
        if not indexSameSeq in seqClass[getClass(labelCurRecordSeq)]: 
            addLabelSeq(record)
    else:  
        addLabelSeq(record)
            

label_seqs = list(label_seqs)
inFile.close()

seqList = label_seqs + none_label_seqs #Phai dung thu tu vi luu nhan dua vao index

#***********Tu file cua Chiba
import itertools

def addLabelSeqChiba(label, seq):
    global seqClass
    global label_seqs
    seqClass[getClass(label)].append(len(label_seqs)) # them index cua chuoi vao dung class
    label_seqs.append(seq)

with open('D:\git_thesis\code\inputfile\chiba_label.txt') as f:
    for label,seq in itertools.izip_longest(*[f]*2):
        label = label.rstrip()
        seq = seq.rstrip()
        if seq in label_seqs:       
            indexSameSeq = label_seqs.index(str(record.seq))  
            # 2 seq giong nhau co cung nhan thi bo, khac nhan thi luu
            if not indexSameSeq in seqClass[getClass(label)]: 
                addLabelSeqChiba(label, seq)
        else:  
            addLabelSeqChiba(label, seq)
        
        
print datetime.now() - startTime



#------------------------------------------------
#
##none_label_seqs =[]
##inFile = open('..//inputfile//HCV_nonlabel.fasta','r')
##
#recordList = []
#recordList1 = []
##for record in SeqIO.parse(inFile,'fasta'):
##    
##    none_label_seqs.append( ''.join(char for char in str(record.seq) if char.isalpha()))
#
#
#label_seqs =[]
#inFile = open('..//inputfile//HCV_label.fasta','r')
#aa = "abc"
#for record in SeqIO.parse(inFile,'fasta'):  
#    if not str(record.seq).isalpha():
#       recordList.append(record)
##       label_seqs.append( ''.join(char for char in str(record.seq) if char.isalpha()))
#    if "AF033375" in str(record.description):
#        recordList1.append(record)
#        
#        
#        
#
##
#thefile = open('test.txt', 'w')
#thefile.writelines(["%s\n\n" % item  for item in recordList])
#
#inFile.close()

