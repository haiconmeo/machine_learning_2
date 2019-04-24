from __future__ import print_function
import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

from scipy import sparse

class uuCF(object):
	def __init__(self ,y_data,k,sim_func=cosine_similarity):
		self.y_data =y_data
		self.k =k
		self.sim_func = sim_func
		self.Ybar =None
		self.n_user = int (np.max(self.y_data[:,0]))+1
		self.n_item =int (np.max(self.y_data[:,1]))+1
	def fit(self):
		users = self.y_data[:,0]
		self.Ybar = self.y_data.copy()
		self.mu = np.zeros((self.n_user,))
		for n in xrange(self.n_user):
			ids = np.where (users == n)[0].astype(np.int32)
			item_ids = self.y_data[ids,1]
			ratings = self.y_data[ids,2]
			self.mu[n] = np.mean(ratings) if ids.size >0 else 0
			self.Ybar[ids,2] = ratings-self.mu[n]
		self.Ybar = sparse.coo_matrix((self.Ybar[:,2],(self.Ybar[:,1],self.Ybar[:,0])),(self.n_item,self.n_user)).tocsr()
		self.S = self.sim_func(self.Ybar.T,self.Ybar.T)
	def pred(self ,u,i):
		ids = np.where(self.y_data[:,1]==i)[0].astype(np.int32)
		users_rated_i = (self.y_data[ids,0]).astype(np.int32)
		sim = self.S[u,users_rated_i]
		nns = np.argsort(sim)[-self.k:]
		neareast_s = sim [nns]
		r = self.Ybar[i,users_rated_i[nns]]
		eps = 1e-8
		return (r*neareast_s).sum()/(np.abs(neareast_s).sum()+eps) + self.mu[u]
r_col = ['user_id','movie_id','rating','unix_timestamp']		
ratings_base =  pd.read_cvs('ml-100k/ua.base',sep='\t',names=r_col)
ratings_test =  pd.read_cvs('ml-100k/ua.test',sep='\t',names=r_col)
rate_train [:,:2] -=1
rate_test [:,:2] -=1

rs = uuCF(rate_train,k=5)
rs.fit()
n_tests = rate_test.shape[0]
SE =0
for n in xrange(n_tests):
	pred = rs.pred(rate_test[n,0],rate_test[n,1])
	SE += (pred-rate_test[n,2])**2
RMSE = np.sqrt(SE/n_tests)	
print (RMSE)
			

		