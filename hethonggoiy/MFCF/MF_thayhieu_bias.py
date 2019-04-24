import numpy as np 
def dem2(k):
	cout=0;
	for i in k:
		if (i-(1e-3))>0.0:
			cout+=1
	return cout		

def MF(Y,K=2,B=0.001,lam=0.03,stop=3000):
	W = np.random.randn(Y.shape[0],K)*10
	#print W
	H = np.random.rand(K,Y.shape[1])*1
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
				#print r
				if r-0<1e-3:
					continue
				
				m= np.sum(Y)/10.0
				
				b_u=0
				b_i=0
				#print dem2(Y[u,:])
				if dem2(Y[u,:])!= 0:
					b_u = np.sum(np.abs(Y[u,:]-m))/(dem2(Y[u,:]))
				#print b_u
				if dem2(Y[:,i])!= 0:
					b_i = np.sum(np.abs(Y[:,i]-m))/(dem2(Y[:,i]))  
					#print ('%.3f,%d,%d,%.3f/%.3f=%.3f'%(m,u,i,np.sum(np.abs(Y[:,i]-m)),dem2(Y[:,i]),b_i))
				
				r_bar=b_u+b_i
				#print ('%.3f,%d,%d,%.3f/%.3f=%3.f'%(m,u,i,np.sum(np.abs(Y[:,i]-m)),np.count_nonzero(Y[:i]),b_i))

				r_bar += np.sum(W[u,:].dot(H[:,i]))
				error =(r-r_bar)
				#print r_bar	

		
				for k in range(K):
					W[u][k]= W[u][k]+B*(error*H[k][i]-lam*np.linalg.norm(W[u][k]))
					H[k][i]= H[k][i]+B*(error*W[u][k]-lam*np.linalg.norm(H[k,i]))
				R_bar=	np.sum(W.dot(H))
				if np.abs(R_bar-R)<1e-4 :
					dem=31111

	return W,H 
Y=np.array([[10,24,0,42],[31,0,0,42],[0,53,0,94],[32,24,0,12]])
X,H=MF(Y)
#print X
	
Y_bar = X.dot(H)	
for i in range(4):
	for j in range(4):
		print ('%.8f  '%Y_bar[i][j],end=''),
	print () 	
