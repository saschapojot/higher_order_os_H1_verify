import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv)!=4:
    print("wrong number of arguments")


group=int(sys.argv[1])
rowNum=int(sys.argv[2])

testNum=int(sys.argv[3])

method_name="S4c"
inDataPath="./combinedData/groupNew"+str(group)+"/row"+str(rowNum)+"/"

inFile=inDataPath+"/test"+str(testNum)+"_"+method_name+"_jsonCombined.txt"


arr=np.loadtxt(inFile)
dtEst = 1e-7
tFlushStart=0
tFlushStop=0.001
flushNum=4000
tTotPerFlush=tFlushStop-tFlushStart
stepsPerFlush=int(np.ceil(tTotPerFlush/dtEst))
dt=tTotPerFlush/stepsPerFlush
# print("dt="+str('{:.2e}'.format(dt)))

tValsAll=np.array(range(1,len(arr)+1))*dt

length=len(arr)

seg=int(length)
# print(np.where(arr>0.8)[0][0])
t2plt=tValsAll[:seg]
arr2plt=arr[:seg]

outDir="./dataJson/groupNew"+str(group)
plt.figure()
plt.plot(t2plt,arr2plt,color="black")
text_x=np.max(t2plt)*1/5
text_y=np.max(arr2plt)*4/5
plt.title("Test"+str(testNum)+": difference between numerical and analytical solution")
plt.xlabel("$t$")
plt.ylabel("diff")
plt.text(text_x, text_y, "$g_{0}=10$, $\omega_{m}=3$, $\omega_{c}=1$, $\omega_{p}=1$", fontsize=12, color='red')
plt.savefig("test"+str(testNum)+"t4.png")
plt.close()