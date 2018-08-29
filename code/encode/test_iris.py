from sklearn.datasets import load_iris
import algorithm_1
import numpy as np
from sklearn.feature_selection import VarianceThreshold


if __name__ == '__main__':
    from datetime import datetime
    startTime = datetime.now()
    data = load_iris() 
    X = data.data
    y = data.target

    for kNumOfNeighbors in range(15,16):
        weightMat = algorithm_1.buildWeightMat(X, [], kNumOfNeighbors)
#        print datetime.now() - startTime
        diagleMat = algorithm_1.buildDiagleMat(weightMat)
        laplaMat = algorithm_1.buildLaplacianMat(weightMat, diagleMat)  
        
        
        featureRowMat = np.transpose(X)
        listLaplacianScore = [algorithm_1.computeLaplacianScore(np.transpose([featureVec]),laplaMat, diagleMat) for featureVec in featureRowMat]
        algorithm_1.saveSortedLaplaFeatureIndexes(listLaplacianScore, filename = ("aaa"))

        
    
    #variance
    sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    new_X = sel.fit_transform(X)