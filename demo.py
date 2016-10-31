'''
Copyleft Oct 30, 2016 Arya Iranmehr, PhD Student, Bafna Lab, UC San Diego,  Email: airanmehr@gmail.com
'''
from CLEAR import *
data=loadSync()
Ts=precomputeTransitions(data)# Ts.to_pickle(path+'T.df');
E,CD=precomputeCDandEmissions(data)
variantScores=HMM(CD, E, Ts)
regionScores=scanGenome(variantScores.alt-variantScores.null,{'CLEAR':lambda x: x.mean()},winSize=200000,step=100000,minVariants=2)
Manhattan(regionScores,std_th=2)
print variantScores
print regionScores
