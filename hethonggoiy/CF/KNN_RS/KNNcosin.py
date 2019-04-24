from __future__ import print_function
import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

def UUCF(Y,func=cosine_similarity):
	return func(Y,Y)
def person(Y):
	mu = np.zeros((4,))
	for i in range(4):
		dem=0
		for j in range(5):
			if Y[i,j] != 0:
				mu[i]+=Y[i,j]
				dem+=1
		mu[i] = float(mu[i])/dem
		for j in range(5):
			if Y[i,j] !=0:
				Y[i,j] = Y[i,j]-mu[i] 
	return Y,mu
def pred(Y,S ,u,i):
	ids = np.where(Y[:,i])[0]
	
	
	
	sim = S[u,ids]
	nns = np.argsort(sim)[-2:]
	#print (nns)
	neareast_s = sim [nns]
	
	r = Y[nns,i]
	#print (r)
	eps = 1e-8
	return (r*neareast_s).sum()/(np.abs(neareast_s).sum()+eps)
#------------------------------------------------------------
def pred2(Y,S ,u,i,m):
	ids = np.where(Y[:,i])[0]
	
	
	
	sim = S[u,ids]
	nns = np.argsort(sim)[-2:]
	#print (nns)
	neareast_s = sim [nns]
	
	r = Y[nns,i]
	#print (r)
	eps = 1e-8
	return (r*neareast_s).sum()/(np.abs(neareast_s).sum()+eps)	+m[u]




#---------------------------------------------------------------	
Y=[[1,4,5,0,3],[5,1,0,5,2],[4,1,2,5,0],[0,3,4,0,4]]

S=UUCF(Y)
print (S)
Y  = np.asarray(Y).astype(np.float)
print (Y)
print ("------------------cosin-------------")
Y2=Y.copy()
Y3=Y.copy()


for u in range(4):
	for i in range(5):
		if Y3[u,i] ==0:
			Y3[u,i] =(pred(Y,S ,u,i))
print (Y3)	
Y,m=	person(Y)
print ("---------------------person----------")
for u in range(4):
	for i in range(5):
		if Y2[u,i] ==0:
			Y2[u,i] =(pred2(Y,S ,u,i,m))
print (Y2)	

