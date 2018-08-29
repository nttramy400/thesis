#import getSeq2
#import numpy as np
#from scipy.spatial.distance import cosine
#import algorithm
#
#if __name__ == '__main__':
##
##    prefixFile = "../inputfile/Data/"
##    nr_isdr_files = ["Chiba/nr_isdr_chiba.txt"]
##    r_isdr_files = ["Chiba/r_isdr_chiba.txt"]
##    
##    #get non_response seq
##    nr_isdr_seqs = getSeq2.getNonResISDRSeq(nr_isdr_files, prefixFile=prefixFile)  #8
##    
##    #get sustained_response seq
##    r_isdr_seqs = getSeq2.getResISDRSeq(r_isdr_files, prefixFile=prefixFile) #20
###    
###    labeled_seqs,indexOfLabelDataDict = combineAllLabeledData(r_isdr_seqs, nr_isdr_seqs)
###    numOfLabeledSeq = len(labeled_seqs)
#    numOfLabeledSeq = 28
#    distMat = np.zeros((numOfLabeledSeq,numOfLabeledSeq))
#    simMat = np.zeros((numOfLabeledSeq,numOfLabeledSeq))
#    nr_isdr_seqs_index = range(139,139+8)
#    r_isdr_seqs_index = range(20)
#    listIndexed = nr_isdr_seqs_index + r_isdr_seqs_index
#    seqRowMat = algorithm.readSeqRowMatFromFile(filename = "D:\git_thesis\code1\outputfile\\10\encodingFile.h5")
#    newSeqRowMat = seqRowMat[listIndexed]
#    
#    for index1 in range(numOfLabeledSeq):
#        for index2 in range(index1,numOfLabeledSeq):
##            distMat[index1][index2] = distMat[index2][index1] = cosine(newSeqRowMat[index1],newSeqRowMat[index2])
#            simMat[index1][index2] = simMat[index2][index1] = 1-cosine(newSeqRowMat[index1],newSeqRowMat[index2])
#            
#            
#
#    import pandas as pd
#    ## convert your array into a dataframe
##    df = pd.DataFrame (distMat)
##    
##    ## save to xlsx file
##    
##    filepath = 'dist_consine_mat.xlsx'
##    
##    df.to_excel(filepath, index=False)
##    
#    df1 = pd.DataFrame (simMat)
#    
#    ## save to xlsx file
#    
#    filepath = 'sim_consine_mat.xlsx'
#    
#    df1.to_excel(filepath, index=False)

import matplotlib.pyplot as plt
import pandas
df = pandas.read_excel(open('sim_consine_mat.xlsx','rb'), sheetname='Sheet1')
df1 = pandas.read_excel(open('D:\\git_thesis\\code1\\blast_sim.xlsx','rb'), sheetname='table')
cos = df.values.reshape(-1)
blast = df1.values

from sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler(feature_range=(0,1))
minMaxScaler.fit(blast)
blast = minMaxScaler.transform(blast).reshape(-1)


plt.plot(range(100), cos[:100], "g-o", label="cosine")
plt.plot(range(100), blast[:100],"r-o", label="blast")
plt.ylabel('similarity')
plt.xlabel('element')
plt.legend()
plt.show()
#plt.savefig("..//outputfile//"+version+ "//"+testVer+"//knnGraph_"+str(k_KNN_feature_selection)+"-knnClassifier_5.png")
plt.close()

    
    
    
    
    
    
    