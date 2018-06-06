import getSeq3
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    version = "11"
    testVer = "1"
    logging.info("------------------GET ISDR SEQUENCES FROM FILE------------------")
    prefixFile = "../inputfile/Data/"
    nr_isdr_files = ["Chiba/nr_isdr_chiba.txt",
                     "Lanl/nr_isdr_lanl.txt",
                     "Paper/nr_isdr_p01.txt", "Paper/nr_isdr_p02.txt","Paper/nr_isdr_p03.txt","Paper/nr_isdr_p04.txt",]
    r_isdr_files = ["Chiba/r_isdr_chiba.txt", 
                    "Lanl/r_isdr_lanl.txt",
                     "Paper/r_isdr_p01.txt", "Paper/r_isdr_p02.txt","Paper/r_isdr_p03.txt","Paper/r_isdr_p04.txt",]
    
    getSeq3.getISDRSeq(nr_isdr_files, r_isdr_files, prefixFile)
    
    logging.info("------------------ENCODING------------------")
