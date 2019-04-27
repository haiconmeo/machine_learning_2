
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

   	

def UUCF(Y,func=cosine_similarity):
	return func(Y,Y)
	

def pred(Y,S ,u,i,k=2):
	ids = np.where(Y[:,i])[0]
	
	
	
	sim = S[u,ids]
	nns = np.argsort(sim)[-k:]
	#print (nns)
	neareast_s = sim [nns]
	
	r = Y[nns,i]
	#print (r)
	eps = 1e-8
	return (r*neareast_s).sum()/(np.abs(neareast_s).sum()+eps)
#------------------------------------------------------------

#---------------------------------------------------------------	
Y=[[1,4,5,0,3],[5,1,0,5,2],[4,1,2,5,0],[0,3,4,0,4]]

#S=UUCF(Y)
S =cosine_similarity(Y).astype(np.float)
Y  = np.asarray(Y).astype(np.float)
print (Y)
print ("------------------cosin-------------")
Y2=Y.copy()
Y3=Y.copy()

print (S) 
print ("------------------user_user-------------")
for u in range(4):
	for i in range(5):
		if Y3[u,i] ==0:
			Y3[u,i] =(pred(Y,S ,u,i))
print (Y3)	
print ("------------------items_items-----------")
Y2=Y2.T
S =cosine_similarity(Y2).astype(np.float)
for i in range(5):
	for u in range(4):
		if Y2[i,u] ==0:
			Y2[i,u] =(pred(Y,S ,i,u))
print (Y2.T)	