import numpy as np
import statistics 
dic= {1:[1,1,1],
	2:[1,2,3],
	3:[2,3,4],
	4:[3,4,5],
	5:[4,5,6],
	6:[5,6,7],
	7:[6,7,8],
	8:[7,8,9],
	9:[9,9,9]}
def doc_file(file):
    lines = []
    with open(file) as fp:
        for line in fp:
            lines.append([int(x) for x in line.split()]) # Used to deal with '\n'
        return lines
def xuly_matran_2(file):
	lines= doc_file(file)
	n=0
	for i in range(len(lines)):
		n = max(lines[i][0],n)
	k= len(lines[1])-2
	# print (k)
	A = np.zeros((n+1,n+1,k))
	for i in range(len(lines)):
		A[lines[i][0]][lines[i][1]]= lines[i][2:]
		
	A_bar=np.zeros((n+1,n+1,3))
	for i in range(n+1):
		for j in range(n+1):
			
			if (A[i][j][0] != 0):
				for t in range(k):
					for x in range(3):
							# print (A[i][j][t])
							 # print (dic[int(A[i][j][t])][k])
						A_bar[i][j][x]+=dic[A[i][j][t]][x]
						

	for i in range(n+1):
		for j in range(n+1):

			if A_bar[i][j][0]>1:
				
				A_bar[i][j]=[x/k for x in A_bar[i][j]]
				A_bar[j][i]=[1/x for x in A_bar[i][j]]
				A_bar[j][i][:] = A_bar[j][i][::-1]






	return A_bar 
def tinh_r(A):
	n = len(A)-1
	r = np.zeros((n+1,3))
	for i in range (n+1):
		b = np.ones(3)
		for j in range (n+1):
			for k in range(3):
				b[k]*=A[i][j][k]
		b[k]=b[k]**(1/(n+1))
		r[i]=b
	return r.astype('float64')
def tinh_w(R):
	S= np.sum(R,axis=0)
	S[:]=[1/x for x in S ]
	S[:]=S[::-1]
	# print (R)
	# print (len(S))
	W=np.ones((len(R),3))
	for i in range(len(R)):
		W[i]=S.T.dot(R[i])
	return W	
def tinh_N(W,R):
	W = np.array(W)
	# print (W.shape)
	R = np.array(R)
	# print (R.shape)
	M=np.ones(len(R))
	for i in range(len(M)) :
		M[i] = W[i].dot(R[i])/3
	s = np.sum(M)	
	M[:] = [x/s for x in M]
	return M


def vectorrieng(A):
	tam1=np.zeros(len(A))
	while(True):
		A=np.array(A)
		x=A.dot(A)
		tam=np.sum(x,axis=1)
		tam/=np.sum(x)
		if (np.sum(tam-tam1)<1e-3):
			break
		tam1=tam
	return tam		
def CI(A):
	ri=[0,0,0.52,0.89,1.11]
	n=len(A)
	# print (n)
	tam1=np.sum(A,axis=0)
	for i in range(len(A)):
		for j in range(len(A)):
			A[i][j]/=tam1[i]
	tam2=np.sum(A,axis=1)
	tam2/=np.sum(A)
	vt=A.dot(tam2)
	tv2=vt/tam2
	print (tv2)
	lamda=np.mean(tv2)
	print (lamda)
	ci=(lamda-n)/(n-1)
	return ci/ri[n-1]


def xuly():
	
	price=xuly_matran_2('PRICE.txt')
	N_price=tinh_N( tinh_w(tinh_r(price)),tinh_r(price))
	distance=xuly_matran_2('DISTANCE.txt')
	N_distance=tinh_N(tinh_w(tinh_r(distance)),tinh_r(distance))
	labor=xuly_matran_2('LABOR.txt')
	N_labor=tinh_N(tinh_w(tinh_r(labor)),tinh_r(labor))
	wage=xuly_matran_2('WAGE.txt')
	N_wage=tinh_N(tinh_w(tinh_r(wage)),tinh_r(wage))
	
	tieuchi=xuly_matran_2('TIEUCHI.txt')
	W=tinh_N(tinh_w(tinh_r(tieuchi)),tinh_r(tieuchi))
	

	m=np.concatenate([(N_price,N_distance,N_labor,N_wage)],axis=1)
	
	print(np.argsort(W.dot(m))[-1])
	
xuly()
