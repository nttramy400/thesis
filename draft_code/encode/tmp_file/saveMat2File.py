# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 00:07:42 2018

@author: BeoU
"""

from datetime import datetime

import numpy as np
import tables

startTime = datetime.now()

sizeOfSeqList =  len(seqList)
# Mo 1 empty HDF5 file moi
fileh = tables.open_file("matrix.h5", mode="w")

# Lay root cua group
root = fileh.root
for feature in setSubString:
    a = np.zeros((sizeOfSeqList,), np.uint16)
    
    #dem so lan xuat hien
    for indexOfSeq in xrange(sizeOfSeqList):
        a[indexOfSeq] = seqList[indexOfSeq].count(feature)
    
    # luu array vao HDF5 file
    hdfarray = fileh.create_array(root, feature, a, feature)

fileh.flush()
fileh.close()

print datetime.now() - startTime #khoang 23 phut

## Open the file for reading
fileh = tables.open_file("matrix.h5", mode="r")
# Get the root group
root = fileh.root

a = root.LL.read()
fileh.close()

#--------------------------------------------------
fileh = tables.open_file("matrix.h5", mode="r")
# Get the root group
root = fileh.root

for array in fileh.walk_nodes("/", "Array"):
    print(array.read())
fileh.close()

#-----------------------------------------------------------------------------
## Open the file for reading
#fileh = tables.open_file("array1.h5", mode="r")
## Get the root group
#root = fileh.root
#
#a = root.array_1.read()
#print("Signed byte array -->", repr(a), a.shape)
#
#print("Testing iterator (works even over scalar arrays):", end=' ')
#arr = root.array_s
#for x in arr:
#    print("nrow-->", arr.nrow)
#    print("Element-->", repr(x))
#    
#b = root.array_s.read()
#
## print "Testing getitem:"
## for i in range(root.array_1.nrows):
##     print "array_1["+str(i)+"]", "-->", root.array_1[i]
#
#print("array_f[:,2:3,2::2]", repr(root.array_f[:, 2:3, 2::2]))
#print("array_f[1,2:]", repr(root.array_f[1, 2:]))
#print("array_f[1]", repr(root.array_f[1]))
#
## Close the file
#fileh.close()
