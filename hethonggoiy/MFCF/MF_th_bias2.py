import numpy as np 

def MF(Y,K=2,B=0.001,lam=0.03,stop=3000):
	W = np.random.randn(Y.shape[0],K)
	#print W
	H = np.random.randn(K,Y.shape[1])
	b= np.random.randn(Y.shape[0])
	d= np.random.randn(Y.shape[1])
	#print H
	U = Y.shape[0]
	I= Y.shape[1]
	R = np.sum(Y)
	#print Y
	dem=0
	while (dem <stop):
		dem+=1
		#print W
		for u in range(U):
			for i in range(I):

				r =Y[u][i]
				if r - 0 <1e-3:
					continue
				#print r
				r_bar=0
				
				r_bar += np.sum(W[u,:].dot(H[:,i]))+b[u]+d[i]
				error =(r-r_bar)
				#print r_bar	
				b[u]=b[u]+B*(error-lam*b[u])
				d[i]=d[i]+B*(error-lam*d[i])

		
				for k in range(K):
					W[u][k]= W[u][k]+B*(error*H[k][i]-lam*np.linalg.norm(W[u][k]))
					H[k][i]= H[k][i]+B*(error*W[u][k]-lam*np.linalg.norm(H[k,i]))
				R_bar=	np.sum(W.dot(H))
				if np.abs(R_bar-R)<1e-3 :
					dem=3111

	return W,H,b,d 
Y=np.array([[10,24,0,42],[31,0,0,42],[0,53,0,94],[32,0,24,12]])
X,H,b,d=MF(Y)
#print X
Y_bar = X.dot(H)+b+d
for i in range(4):
	for j in range(4):
		print ('%.8f'%Y_bar[i][j]),
	print ()	
			