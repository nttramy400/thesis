import tables
import logging
logging.basicConfig(level=logging.DEBUG)
############################ READ INFO FROM H5 FILE ############################
def getSeqListFromH5File(filename= "info.h5"):
    logging.info("Get all raw ISDR sequences from .h5 file")
    infoFile = tables.open_file(filename, mode = 'r')
    seqList = infoFile.root.sequences.read()
    infoFile.close()
    return seqList

def readFeatureRowMatFromFile(filename="encodingFile.h5"):
    logging.info("Get relational matrix which has rows are features from .h5 file")
    fileh = tables.open_file(filename, mode="r")        
    featureRowMatrix = fileh.root.featureRowMatrix.read()
    fileh.close()
    return featureRowMatrix
    
def readSeqRowMatFromFile(filename="encodingFile.h5"):
    logging.info("Get relational matrix which has rows are sequences from .h5 file")
    fileh = tables.open_file(filename, mode="r")        
    seqRowMatrix = fileh.root.seqRowMatrix.read()
    fileh.close()
    return seqRowMatrix

def readFeatureListFromFile(filename="encodingFile.h5"):
    logging.info("Get all raw features from .h5 file")
    fileh = tables.open_file(filename, mode="r")        
    featureList = fileh.root.feature.read()
    fileh.close()
    return featureList

def readLabelVecFromFile(filename="info.h5"):
    logging.info("Get vetor of labeled from .h5 file")
    fileh = tables.open_file(filename, mode="r")
    labelVec = fileh.root.labelVec.read()
    fileh.close()
    return labelVec
