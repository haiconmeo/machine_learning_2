import numpy as np
def doc_file(file):
    lines = []
    with open(file) as fp:
        for line in fp:
            lines.append([int(x) for x in line.split()]) # Used to deal with '\n'
        return lines

def xuly_matran(file):
	lines= doc_file(file)
	n=0
	for i in range(len(lines)):
		n = max(lines[i][0],n)
	
	
	A = np.zeros((n+1,n+1))
	for i in range(len(lines)):
		A[lines[i][0]][lines[i][1]] = lines[i][2]
		A[lines[i][1]][lines[i][0]] = float(1/lines[i][2])
	for i in range(n+1):
		for j in range(n+1):
			if A[i][j] == 0:
				if i ==j :
					A[i][j] =1
				else:	
					for k in range(len(lines)):
						if (A[i][k]!=0 and A[k][j]!=0):
							A[i][j]=A[i][k]*A[k][j]
							A[j][i]=float(1/A[i][j])
							break

    
	return A    
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
	print (n)
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
	
	price=xuly_matran('PRICE.txt')
	vectorrieng_price=vectorrieng(price)
	distance=xuly_matran('DISTANCE.txt')
	vectorrieng_distance=vectorrieng(distance)
	labor=xuly_matran('LABOR.txt')
	vectorrieng_labor=vectorrieng(labor)
	wage=xuly_matran('WAGE.txt')
	vectorrieng_wage=vectorrieng(wage)
	
	tieuchi=xuly_matran('TIEUCHI.txt')
	vectorrieng_tieuchi=vectorrieng(tieuchi)
	

	m=np.concatenate([(vectorrieng_price,vectorrieng_distance,vectorrieng_labor,vectorrieng_wage)],axis=1)
	
	print(vectorrieng_tieuchi.dot(m))
	print (CI(price))
	
xuly()	
