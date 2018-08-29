import h5filereader
import numpy as np

seqRowMatrix  = h5filereader.readSeqRowMatFromFile()
labelVec = h5filereader.readLabelVecFromFile()
featureName = h5filereader.readFeatureListFromFile()


numOfLabel = len(labelVec)

labelSeqRowMatrix = seqRowMatrix[:numOfLabel]
unlabelSeqRowMatrix = seqRowMatrix[numOfLabel:]

np.save("labelVec",labelVec)
np.save("labelMat",labelSeqRowMatrix)
np.save("unlabelMat",unlabelSeqRowMatrix)
np.save("featureName",featureName)