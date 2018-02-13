from datetime import datetime
import tables

startTime = datetime.now()


fileh = tables.open_file("relationalMat.h5", mode="r")
print datetime.now() - startTime 


root = fileh.root

LL = root.LL
print (LL._v_attrs)
#root._v_nchildren: the num of children

arr = root._f_get_child('LL').read()

    
fileh.close()


