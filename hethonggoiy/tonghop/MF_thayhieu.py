import numpy as np 

def MF(Y,K=2,B=0.001,lam=0.03,stop=3000):
	W = np.random.randn(Y.shape[0],K)
	#print W
	H = np.random.randn(K,Y.shape[1])
	#print H
	U = Y.shape[0]
	I = Y.shape[1]
	R = np.sum(Y)
	#print Y
	dem=0
	while (dem <stop):
		dem+=1
		#print W
		for u in range(U):
			for i in range(I):

				r =Y[u][i]
				if r ==0:
					continue
				#print r
				r_bar=0
				
				r_bar += np.sum(W[u,:].dot(H[:,i]))
				error =(r-r_bar)
				#print r_bar	

		
				for k in range(K):
					W[u][k]= W[u][k]+B*(error*H[k][i]-lam*np.linalg.norm(W[u][k]))
					H[k][i]= H[k][i]+B*(error*W[u][k]-lam*np.linalg.norm(H[k,i]))
				R_bar=	np.sum(W.dot(H))
				e=0
				for i in range(U):
					for j in range(I):
						if Y[i][j] > 0:
							e = e + pow(Y[i][j] - W[u,:].dot(H[:,i]), 2)
							for k in range(K):
								e = e + (lam/2) * ( pow(W[i][k],2) + pow(H[k][j],2) )
								if e < 0.001:
									break

	return W,H 
Y=np.array([[10,24,0,42],[31,0,0,42],[0,53,0,94],[32,0,24,12]])
X,H=MF(Y)
#print X
Y_bar = X.dot(H)	
for i in range(4):
	for j in range(4):
		print ('%.8f   '%Y_bar[i][j]  ,end='')
	print ()