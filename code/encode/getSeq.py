import tables
version = "8"
def getListNonLabelSeq(filenameList = ["..//inputfile//unlabeled_data.txt"]):
    seqList = []
    for fname in filenameList:
        with open(fname) as f:
            content = f.readlines()
            seqList+=content
    return [x.strip() for x in seqList] 
#    return list(set([x.strip() for x in seqList])) 

  

def getInfoLabelSeq(filenameList = ["..//inputfile//labeled_data.txt"]):
    seqList = []
    indexOfLabelDataDict = {} #luu index cua sequence tung loai
    for fname in filenameList:
        with open(fname) as f:
            contentList = f.readlines()
            for content in contentList:
                seq, label = content.split()
                
                #check duplicate
                if False:
                    pass
#                if ((label in indexOfLabelDataDict) and (seq in [seqList[index] for index in indexOfLabelDataDict[label]])): #bi duplicate (giong ca seq va nhan)
#                    pass
                else: #khong duplicate
                    #them sequence vao list
                    seqList.append(seq)
                    
                    #them label vao dict
                    if label in indexOfLabelDataDict: #neu nhan da xhien trong dictionary, them index seq vao list ung voi nhan do
                        indexOfLabelDataDict[label].append(len(seqList)-1)
                    else: #label chua xuat hien trong dict => tao list moi
                        indexOfLabelDataDict[label]= [len(seqList)-1]
    return (seqList,indexOfLabelDataDict)
    
def saveInfoSeqToH5File(seqList,indexOfLabelDataDict, filename = "..//outputfile//"+version+"//info.h5"):
    fileh = tables.open_file(filename,'w')
    root = fileh.root
    labelGroup = fileh.create_group(root, "label")
    
    for key in indexOfLabelDataDict:
        fileh.create_array(labelGroup, key, indexOfLabelDataDict[key], key)
    
    fileh.create_array(root, 'sequences', seqList, 'sequences')
    
    fileh.flush()
    fileh.close()
                
if __name__ == '__main__':
    labeledSeq, indexOfLabelDataDict = getInfoLabelSeq()
    nonLabeledSeq = getListNonLabelSeq()
    seqList = labeledSeq + nonLabeledSeq
    saveInfoSeqToH5File(seqList, indexOfLabelDataDict)