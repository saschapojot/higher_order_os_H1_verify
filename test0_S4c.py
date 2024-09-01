import numpy as np
import pandas as pd
from datetime import datetime
import sys
from scipy.special import hermite
import copy
import pickle
from pathlib import Path
from scipy import sparse
import json



#this script tests accuracy for S4c

# python readCSV.py groupNum rowNum, then parse csv
if len(sys.argv)!=4:
    print("wrong number of arguments")


group=int(sys.argv[1])
rowNum=int(sys.argv[2])

testNum=int(sys.argv[3])
inParamFileName="./inParamsNew"+str(group)+".csv"

#read parameters from csv
dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]



g0=float(oneRow.loc["g0"])
omegam=float(oneRow.loc["omegam"])
omegap=float(oneRow.loc["omegap"])
omegac=float(oneRow.loc["omegac"])
er=float(oneRow.loc["er"])#magnification
r=np.log(er)
thetaCoef=float(oneRow.loc["thetaCoef"])
theta=thetaCoef*np.pi
Deltam=omegam-omegap
e2r=er**2
lmd=(e2r-1/e2r)/(e2r+1/e2r)*Deltam


N2=2
L1=5
L2=80
N1=2

dx1=2*L1/N1
dx2=2*L2/N2

print("dx1="+str(dx1))
print("dx2="+str(dx2))


x1ValsAll=np.array([-L1+dx1*n1 for n1 in range(0,N1)])
x2ValsAll=np.array([-L2+dx2*n2 for n2 in range(0,N2)])
x1ValsAllSquared=x1ValsAll**2
x2ValsAllSquared=x2ValsAll**2


#time-independent part of H1
H1_space=np.zeros((N1,N2),dtype=complex)

for n1 in range(0,N1):
    for n2 in range(0,N2):
        x1SqTmp=x1ValsAllSquared[n1]
        x2Tmp=x2ValsAll[n2]
        H1_space[n1,n2]=g0*omegac*np.sqrt(2*omegam)*x1SqTmp*x2Tmp-1/2*g0*np.sqrt(2*omegam)*x2Tmp

#to construct psi analytical
matSpace=np.zeros((N1,N2),dtype=complex)

for n1 in range(0,N1):
    for n2 in range(0,N2):
        x1SqTmp=x1ValsAllSquared[n1]
        x2Tmp=x2ValsAll[n2]
        matSpace[n1,n2]=1/2*g0*np.sqrt(2*omegam)*x2Tmp-g0*omegac*np.sqrt(2*omegam)*x1SqTmp*x2Tmp

matSpace*=1j/omegap


def psiAnalytical(t):
    psiTmp=np.exp(matSpace*np.sin(omegap*t))
    psiTmp/=np.linalg.norm(psiTmp,"fro")
    return psiTmp




dtEst = 1e-4
tFlushStart=0
tFlushStop=0.001
flushNum=4000
tTotPerFlush=tFlushStop-tFlushStart

stepsPerFlush=int(np.ceil(tTotPerFlush/dtEst))
dt=tTotPerFlush/stepsPerFlush

timeValsAll=[]
for fls in range(0,flushNum):
    startingInd = fls * stepsPerFlush
    for j in range(0,stepsPerFlush):
        timeValsAll.append(startingInd+j)


timeValsAll=np.array(timeValsAll)*dt
outDir="./groupNew"+str(group)+"/row"+str(rowNum)+"/test"+str(testNum)+"S4c_H1Verify/"
Path(outDir).mkdir(parents=True, exist_ok=True)

def evolution1Step(j,psi):
    """

    :param j: time step
    :param psi: wavefunction at the beginning of the time step j
    :return:
    """
    tj=timeValsAll[j]
    #S4c
    #propagator 1
    U1=np.exp(-1j*1/6*dt*H1_space*np.cos(omegap*tj))
    psi_1=U1*psi

    #propagator 2
    U2=np.exp(-1j*2/3*dt*H1_space*np.cos(omegap*(tj+1/2*dt)))
    psi_2=U2*psi_1

    #propagator 3
    U3=np.exp(-1j*1/6*dt*H1_space*np.cos(omegap*(tj+dt)))
    psi_3=U3*psi_2

    return psi_3

tEvoStart=datetime.now()

psiNumericalCurr=psiAnalytical(0)

psiAnaCurr=psiAnalytical(0)


for fls in range(0,flushNum):
    tFlushStart = datetime.now()
    startingInd = fls * stepsPerFlush
    diffPerFlush=[]
    for st in range(0,stepsPerFlush):
        j = startingInd + st
        psiNumericalNext = evolution1Step(j, psiNumericalCurr)
        psiNumericalCurr = psiNumericalNext
        psiAnaCurr = psiAnalytical(timeValsAll[j] + dt)
        diffTmp = np.linalg.norm(psiNumericalCurr - psiAnaCurr, ord="fro")
        diffPerFlush.append(diffTmp)

    outData = {"diff": diffPerFlush}
    outFile = outDir + "flush" + str(fls) + "N1" + str(N1) \
              + "N2" + str(N2) + "L1" + str(L1) \
              + "L2" + str(L2) + "diff.json"

    with open(outFile,"w") as fptr:
        json.dump(outData,fptr)

    tFlushEnd = datetime.now()
    print("flush "+str(fls)+" time: ",tFlushEnd-tFlushStart)


tEvoEnd=datetime.now()
print("evo time: ",tEvoEnd-tEvoStart)