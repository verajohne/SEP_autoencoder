import numpy as np
import scipy as sp
from numpy.random import RandomState
import pandas as pd

from scipy.stats import norm
#from MVN import MVN
from scipy.linalg import det
import gauss as gauss

import seaborn as sns


RS = RandomState(1213)
np.random.seed(42)

class FAnalysis(object):
	
	def __init__(self,n, dimz = 2, dimx = 3):
		
		self.n = n
		self.W = RS.normal(0,1, size = (dimx,dimz))
		self.sigx = 0.0000000000000001#RS.normal(0,1)
		self.dimz = dimz
		self.dimx = dimx
		
		data = self.generate_data(n)
		self.observed = data[0]
		#we keep this to test
		self.latent = data[1]
		
		self.prec = (1/self.sigx)*np.dot(self.W.transpose(), self.W)
		self.cov = np.linalg.inv(self.prec)
		
		'''
		values for normalisation computation
		'''
		temp1 = (2*np.pi)**(dimz/2.0)*np.sqrt(det(self.cov))
		temp2 = det(2*np.pi*self.sigx*np.identity(dimz))
		self.pc_norm1 = temp1/temp2
		temp3 = np.linalg.inv(np.dot(self.W.transpose(), self.W))
		self.wtwinv = temp3
		temp3 = np.dot(self.W, temp3)
		self.pc_norm2 = np.dot(temp3, self.W.transpose())
		
		'''
		prior for n factors will be product of n priors~ N(0,I)
		'''
		#I = np.identity(dimz)
		#product_priors = MVN(0, n*I)
		'''
		factors- initialised to ~N(0,1) 
		'''
		#product_factors = MVN(0, n*I)
		
		#self.aprx_posterior = None
		
		
	def generate_data(self,n):
		observed = np.zeros([n,self.dimx])
		latent = np.zeros([n, self.dimz])
	
		for i in xrange(n):
			#latent variable
			z = np.random.normal(0,1, size = (self.dimz,))

			#observed
			mu = np.dot(self.W,z)
			cov = self.sigx*np.identity(3)
			x = np.random.multivariate_normal(mu, cov)
		
			observed[i] = x
			latent[i] = z

		return observed, latent
	
	def get_normalisation_constant(self, x):
		a = 1/(2*self.sigx)
		temp = - a*np.dot(x.transpose(), x) + a*np.dot(np.dot(x.transpose(), self.pc_norm2), x)
		return self.pc_norm1*np.exp(temp)
		
	def get_mu(self, x):
		W = self.W
		temp = np.dot(self.W.transpose(), W)
		temp = np.linalg.inv(temp)
		temp = np.dot(temp, W.transpose())
		
		return np.dot(temp, x)
		
	def EP(self):
		norm = 1
		mus = np.array([])
		norms = []
		for i in xrange(self.n):
			mu = self.get_mu(self.observed[i])
			nc = self.get_normalisation_constant(self.observed[i])
			norm *= nc
			norms.append(nc)
			mus = np.hstack((mus, mu))
		mus = mus.reshape((10,2))
		sig = np.dot(self.W.transpose(), self.W)
		sig = sig/self.sigx
		sig = np.linalg.inv(sig)
		return sig, mus, norms
		
	def plot_true_posterior(self,index):
		x = self.observed[index].transpose()
		print self.latent[index]
		a = self.sigx*np.identity(3)
		temp = np.identity(2) + np.dot(np.dot(self.W.transpose(), np.linalg.inv(a)), self.W)
		temp = np.linalg.inv(temp)
		temp = np.dot(temp, self.W.transpose())
		temp = np.dot(temp, np.linalg.inv(a))
		mean = np.dot(temp,x)
		temp = np.dot(self.W.transpose(), np.linalg.inv(a))
		cov = np.linalg.inv(np.identity(2) + np.dot(temp, self.W))
		
		print mean
		print""
		print cov
		
		data = np.random.multivariate_normal(mean, cov, 20000)
		#data2 = np.random.multivariate_normal(mean, cov, 5)
		#print "Data: ", data == data2
		
		
		df = pd.DataFrame(data, columns=["x", "y"])
		#sns.jointplot(x="x", y="y", data=df)
		xlim = (mean[0] - 3*np.sqrt(cov[0][0]),mean[0] + 3*np.sqrt(cov[0][0]))
		ylim = (mean[1] - 3*np.sqrt(cov[1][1]),mean[1] + 3*np.sqrt(cov[1][1]))
		sns.jointplot(x="x", y="y", data=df, kind="kde", xlim = xlim, ylim = ylim)
		
		epcov, epmeans, norms = self.EP()
		epmean, epcov = gauss.prod_gauss(epmeans[index], np.zeros(2), epcov, np.identity(2))
		#print mean == epmean
		#print epcov == cov
		print sp.linalg.norm(mean - epmean)
		print sp.linalg.norm(cov - epcov)
		
		data = RS.multivariate_normal(epmean, epcov, 20000)

		df = pd.DataFrame(data, columns=["x", "y"])
		#sns.jointplot(x="x", y="y", data=df)
		xlim = (epmean[0] - 3*np.sqrt(epcov[0][0]),epmean[0] + 3*np.sqrt(epcov[0][0]))
		ylim = (epmean[1] - 3*np.sqrt(epcov[1][1]),epmean[1] + 3*np.sqrt(epcov[1][1]))
		sns.jointplot(x="x", y="y", data=df, kind="kde", xlim = xlim, ylim = ylim)

	def plot_approx_posterior(self, index):
		cov, means, norms = self.EP()
		mean = means[index]
		print mean.shape
		mean, cov = gauss.prod_gauss(mean, np.zeros(2), cov, np.identity(2))
		print mean
		print cov
		data = np.random.multivariate_normal(mean, cov, 200)
		df = pd.DataFrame(data, columns=["x", "y"])
		#sns.jointplot(x="x", y="y", data=df)
		xlim = (mean[0] - 3*np.sqrt(cov[0][0]),mean[0] + 3*np.sqrt(cov[0][0]))
		ylim = (mean[1] - 3*np.sqrt(cov[1][1]),mean[1] + 3*np.sqrt(cov[1][1]))
		sns.jointplot(x="x", y="y", data=df, kind="kde", xlim = xlim, ylim = ylim)		
		






if __name__ == "__main__":
	fa = FAnalysis(10)
	sig, mu, norms = fa.EP()
	fa.plot_approx_posterior(0)
	#fa.plot_true_posterior(0)
	#plot(0, mu, sig, norms, fa.latent)
		
	
	
	
	
	
	
	
	
	
	