from __future__ import print_function
import pandas as pd 
import numpy as np 
#from sklean.metrics,pairwise import consine_similarity
from scipy import sparse

class MFCF(object):
	"""docstring for MFCF"""
	def __init__(self, Y,K,lam=0.1,Xint=None,Wint=None,learning_rate=0.5,max_iter=1000,print_every=100):
		self.Y= Y
		self.K= K
		self.lam = lam
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.print_every = print_every
		self.n_users =  int(np.max(Y[:,0]))+1
		self.n_items = int(np.max(Y[:,1]))+1
		self.n_rating = Y.shape[0]
		self.X = np.random.randn(self.n_items,K) if Xint is None else Xint
		self.W = np.random.randn(K,self.n_users) if Wint is None else Wint
		self.b = np.random.randn(self.n_items)
		self.d = np.random.randn(self.n_users)

		
	def loss(self):
		L =0
		for i in range(self.n_rating):
			n,m,rating =  int(self.Y[i,0]),int(self.Y[i,1]),self.Y[i,2]
			L+=0.5*(self.X[m].dot(self.W[:,n])+self.b[m]+self.d[n]-rating)**2
		L/=self.n_rating
		return L+0.5*self.lam*(np.sum(self.X**2)+np.sum(self.W**2))	
		#return L+0.5*self.lam*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro') + \
          #                np.linalg.norm(self.b) + np.linalg.norm(self.d))
	def updateXb(self):
		for m in range(self.n_items):
			ids = np.where(self.Y[:,1]==m)[0]
			user_ids,ratings = self.Y[ids,0].astype(np.int32),self.Y[ids,2]
			Wm,dm = self.W[:,user_ids],self.d[user_ids]
			
			xm = self.X[m]
			error = xm.dot(Wm)+self.b[m]+dm-ratings
			grad_xm = error.dot(Wm.T)/self.n_rating+self.lam*xm
			grad_bm= np.sum(error)/self.n_rating
			self.X[m]-=self.learning_rate*grad_xm.reshape(-1)
			self.b[m]-=self.learning_rate*grad_bm
	def updateWd(self):
		for n in range(self.n_users):
			ids = np.where(self.Y[:,0]==n)[0]
			items_ids,ratings = self.Y[ids,1].astype(np.int32),self.Y[ids,2]
			Xn,bn = self.X[items_ids],self.b[items_ids]
			
			wn = self.W[:,n]
			error = Xn.dot(wn)+bn+self.d[n]-ratings
			grad_wn = (Xn.T.dot(error))/self.n_rating+self.lam*wn
			grad_dn= np.sum(error)/self.n_rating
			self.W[:,n]-=self.learning_rate*grad_wn.reshape(-1)
			self.b[n]-=self.learning_rate*grad_dn
	def fit(self):

		for it in range(self.max_iter):
			self.updateWd()
			self.updateXb()
			if(it+1)% self.print_every==0:
				rmse_train = self.evaluate_RMSE(self.Y)
				print('iter=%d,loss =%.4f,RMSE train =%.4f' %(it+1,self.loss(),rmse_train))
	def pred(self,u,i):
		u,i=int(u),int(i)
		pred= self.X[i,:].dot(self.W[:,u])+self.b[i]+self.d[u]
		return max(0,min(5,pred))
	def evaluate_RMSE(self,rate_test):
		n_tests = rate_test.shape[0]
		SE = 0
		for n in range(n_tests):
			pred = self.pred(rate_test[n,0],rate_test[n,1])
			SE+=(pred-rate_test[n,2])**2
		RMSE =np.sqrt(SE/n_tests)	
		return RMSE
r_col = ['user_id','movie_id','rating','unix_timestamp']		
ratings_base =  pd.read_csv('ml-100k/ua.base',sep='\t',names=r_col)
ratings_test =  pd.read_csv('ml-100k/ua.test',sep='\t',names=r_col)
rate_train=ratings_base.as_matrix()
rate_test = ratings_test.as_matrix()
rate_train [:,:2] -=1
rate_test [:,:2] -=1
rs = MFCF(rate_train,K=50,lam=0.1,print_every=5,learning_rate=50,max_iter=30)	
rs.fit()	
RMSE = rs.evaluate_RMSE(rate_test)
print('RMSE=%.4f'%RMSE)

					

