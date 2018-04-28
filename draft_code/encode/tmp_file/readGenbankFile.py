# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 02:46:15 2018

@author: BeoU
"""

from Bio import SeqIO
gb_file = "AF033366.gbf"
record = "abc"
fileh = "/"
#fileh = open(gb_file,"r")
for gb_record in SeqIO.parse(open(gb_file,"r"), "genbank") :
    # now do something with the record
    print "Name %s, %i features" % (gb_record.name, len(gb_record.features))
    print repr(gb_record.seq)
    record = gb_record
#    
    
from Bio import SeqIO
gb_file = "D:\git_thesis\code\encode\HCV2.genbank"
gb_record = SeqIO.read(open(gb_file,"r"), "genbank")
print "Name %s, %i features" % (gb_record.name, len(gb_record.features))
print repr(gb_record.seq)