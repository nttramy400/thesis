import itertools
import tables
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

def getSeqFromTextFile(filenameList):
    if (filenameList == None):
        return []
    logging.info("Get sequences from file which don't have fasta format")
    seqList = []
    for fname in filenameList:
        with open(fname) as f:
            content = f.readlines()
            seqList+=content
#    return list(set([x.strip() for x in seqList])) 
    return [x.strip() for x in seqList] 

def getNonResISDRSeq(nr_isdr_files):
    logging.info("Get non-response ISDR sequences")
    nr_isdr_seqs=[]
    for filename in nr_isdr_files:
        with open(filename) as f:
            for name,seq in itertools.izip_longest(*[f]*2):
    #            name = name.rstrip()
                seq = seq.rstrip()
                nr_isdr_seqs.append(seq)
    return nr_isdr_seqs
    

def getResISDRSeq(r_isdr_files):
    logging.info("Get sustained_response ISDR sequences")
    r_isdr_seqs=[]
    for filename in r_isdr_files:
        with open(filename) as f:
            for name,seq in itertools.izip_longest(*[f]*2):
    #            name = name.rstrip()
                seq = seq.rstrip()
                r_isdr_seqs.append(seq)
    return r_isdr_seqs       

def getListNonLabelSeq(filenameList):
    logging.info("Get non-labeled ISDR sequences")
    seqList = []
    for fname in filenameList:
        with open(fname) as f:
            content = f.readlines()
            seqList+=content
#    return list(set([x.strip() for x in seqList])) 
    return [x.strip() for x in seqList] 

def saveInfoSeqToH5File(seqList,labelVec, filename = "info.h5"):
    logging.info("Save all ISDR sequences to file")
    fileh = tables.open_file(filename,'w')
    root = fileh.root
    
    fileh.create_array(root, 'sequences', seqList, 'sequences')
    fileh.create_array(root, 'labelVec', labelVec, 'labelVec')
    
    fileh.flush()
    fileh.close()
    
def combineAllLabeledData(r_isdr_seqs, nr_isdr_seqs):
    logging.info("Combine all ISDR sequences list")
#    #neu can remove cac chuoi giong nhau trong cung 1 nhan
#    nr_isdr_seqs = list(set(nr_isdr_seqs))
#    r_isdr_seqs = list(set(r_isdr_seqs)) 
    
    
#    indexOfLabelDataDict = {"sustained_response":[], "non_response":[]} #luu index cua sequence tung loai
    seqList = r_isdr_seqs + nr_isdr_seqs
    numOfResSeq = len(r_isdr_seqs)
    numOfNonResSeq = len(nr_isdr_seqs)
    numOfLabeledSeq = numOfResSeq + numOfNonResSeq
    labelVec = np.empty(numOfLabeledSeq, dtype = 'S18')
    #order of labelVec based on order of seqList
    for i in range(numOfResSeq):
        labelVec[i] = "sustained_response"
    for i in range(numOfResSeq,numOfLabeledSeq):
        labelVec[i] = "non_response"
    return seqList,labelVec



def removeSeqMultiLabel(nr_isdr_seqs, r_isdr_seqs):
    logging.info("Remove sequences have multiple label")
        #xu ly chuoi nam o ca 2 label
    duplabel_isdr_seqs = list(set(nr_isdr_seqs).intersection(r_isdr_seqs)) 
    
    #co duplicate
#    for seq in duplabel_isdr_seqs:
#        freq_nr_isdr = nr_isdr_seqs.count(seq)
#        freq_r_isdr = r_isdr_seqs.count(seq)
#        if freq_nr_isdr < freq_r_isdr:
#            nr_isdr_seqs.remove(seq)
#        elif freq_nr_isdr > freq_r_isdr:
#            r_isdr_seqs.remove(seq)
#        else: #==
#            pass

         
    
def getISDRSeq(nr_isdr_files, r_isdr_files, nonlabel_isdr_files):
    logging.info("Get ISDR sequences from files")
    #get non_response seq   
    nr_isdr_seqs = getSeqFromTextFile(nr_isdr_files)
    
    #get sustained_response seq
    r_isdr_seqs = getSeqFromTextFile(r_isdr_files)
    
    #get non_labeled_seq
    non_label_isdr_seqs = getSeqFromTextFile(nonlabel_isdr_files)
    
    labeled_seqs,labelVec = combineAllLabeledData(r_isdr_seqs, nr_isdr_seqs)
    
    seqList = labeled_seqs + non_label_isdr_seqs
    
    saveInfoSeqToH5File(seqList, labelVec)
    
if __name__ == '__main__':
    prefixFile = "D:\\git_thesis\\code1\\inputfile\\Data\\"
    nr_isdr_files = ["Chiba/nr_isdr_chiba.txt",
                     "Lanl/nr_isdr_lanl.txt",
                     "Paper/nr_isdr_p01.txt", "Paper/nr_isdr_p02.txt","Paper/nr_isdr_p03.txt","Paper/nr_isdr_p04.txt",]
    r_isdr_files = ["Chiba/r_isdr_chiba.txt", 
                    "Lanl/r_isdr_lanl.txt",
                     "Paper/r_isdr_p01.txt", "Paper/r_isdr_p02.txt","Paper/r_isdr_p03.txt","Paper/r_isdr_p04.txt",]
    nr_isdr_files = [prefixFile+filename for filename in nr_isdr_files]
    r_isdr_files = [prefixFile+filename for filename in r_isdr_files]
    
    #save after parsing sequences
    nr_isdr_seqs = getNonResISDRSeq(nr_isdr_files)
    r_isdr_seqs = getResISDRSeq(r_isdr_files)
    
    outfile = open("nr_isdr.txt","w")
    for seq in nr_isdr_seqs:
        outfile.write(seq + "\r\n")
        
    outfile.flush()
    outfile.close()
    
    outfile = open("r_isdr.txt","w")
    for seq in r_isdr_seqs:
        outfile.writelines(seq + "\r\n")
        
    outfile.flush()
    outfile.close()
    


    
    
    
    
    
    
    

