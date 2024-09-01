import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import json
import sys
from pathlib import Path


if len(sys.argv)!=4:
    print("wrong number of arguments")


group=int(sys.argv[1])
rowNum=int(sys.argv[2])

testNum=int(sys.argv[3])

inJsonFileNames=[]
flushNumAll=[]

method_name="S4c"

dataPath="./groupNew5/row"+str(rowNum)+"/test"+str(testNum)+method_name+"_H1Verify/"
for file in glob.glob(dataPath+"/*.json"):

    matchFlush=re.search(r"flush(\d+)N1",file)
    if matchFlush:
        flushNumAll.append(int(matchFlush.group(1)))
        inJsonFileNames.append(file)
sortedInds=np.argsort(flushNumAll)
sortedFileNames=[inJsonFileNames[ind] for ind in sortedInds]

# print(sortedFileNames)
diffValsAll=np.array([])
for file in sortedFileNames:
    with open(file,"r") as fptr:
        dataTmp=json.load(fptr)
    diffOneFile=np.array(dataTmp["diff"])
    diffValsAll=np.r_[diffValsAll,diffOneFile]

outDataPath="./combinedData/groupNew"+str(group)+"/row"+str(rowNum)+"/"
Path(outDataPath).mkdir(exist_ok=True, parents=True)
np.savetxt(outDataPath+"/test"+str(testNum)+"_"+method_name+"_jsonCombined.txt",diffValsAll,delimiter=",")