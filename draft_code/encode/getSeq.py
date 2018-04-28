from datetime import datetime  
from Bio import SeqIO
import tables
startTime = datetime.now()
#---------------------------- GET SEQUENCES FROM FILES -----------------------
######################################## KHONG NHAN ########################################
none_label_seqs = set() #loai cac seq giống nhau


inFile = open('..//inputfile//HCV_nonlabel.txt','r')

for record in SeqIO.parse(inFile,'fasta'):
    if 'ns5a' in record.description.lower():
        none_label_seqs.add(str(record.seq))

none_label_seqs = list(none_label_seqs)
inFile.close()


######################################## CO NHAN ########################################
#***********************Tu file fasta LANL********************************************
#co 2 class la sustained_response, non-response
labelClass = {"sustained_response":["sustained_response", "SVR"],
              "non_response":["non-response", "relapse", "Null"]}
seqClass = {"sustained_response":[], "non_response":[]} #luu index cua sequence tung loai

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

def addLabelSeqChiba(label, seq):
    global seqClass
    global label_seqs
    seqClass[getClass(label)].append(len(label_seqs)) # them index cua chuoi vao dung class
    label_seqs.append(seq)
    

def getSeqByLengtth(length = 447):
    global none_label_seqs
    global label_seqs
    none_label_seqs = filter(lambda seq: len(seq)==length, none_label_seqs)
    nonProteinLabels = filter(lambda seq: len(seq[1])!=length, enumerate(label_seqs))
    nonProteinLabelIndexes = [x[0] for x in nonProteinLabels] # chi lay index
    for key in seqClass:
        seqClass[key] = list(set(seqClass[key]) - set(nonProteinLabelIndexes))
        
    for index in sorted(nonProteinLabelIndexes, reverse=True):
        del label_seqs[index]

label_seqs = []

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


#***********************Tu file cua Chiba*********************************************
import itertools



with open('..//inputfile//chiba_label.txt') as f:
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
        
##########cong don thanh 1 list sequence
getSeqByLengtth(447) # chi lay cac seq co length = 447
seqList = label_seqs + none_label_seqs #Phai dung thu tu vi luu nhan dua vao index
seqList = map(lambda seq: seq[236:276], seqList) #lay ISDR

print datetime.now() - startTime



#------------------------------------------------------------------------------------
#Save summary to file
fileh = tables.open_file('..//outputfile//info.h5','w')
root = fileh.root
labelGroup = fileh.create_group(root, "label")

for key in seqClass:
    fileh.create_array(labelGroup, key, seqClass[key], key)

fileh.create_array(root, 'sequences', seqList, 'sequences')

fileh.flush()
fileh.close()
