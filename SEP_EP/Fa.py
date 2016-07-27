import sys
sys.path.append('../')

from utils import util
from exact import exact

import numpy as np

class Fa(object):
	
	def __init__(self, n, dimx, dimz):
	
		self.n = n
		self.sigx = 0.0001
		sigw = np.random.normal(0,1)
		self.W = np.random.normal(0,sigw, size = (dimx,dimz))
		
		#data
		data = util.generate_data(n, self.W, self.sigx, dimx, dimz)
		self.observed = data[0]
		self.latent = data[1]
		
		#SEP params: (means, exponent values of covariance matrix (e^-b) - b on the diagonal)
		f = dimx*dimz
		self.SEP_prior = (np.zeros(f), np.ones(f))
		self.SEP_factor = (np.zeros(f), np.zeros(f)))
	
	def sample_theta(self):
		pass
	
	def evaluate_joing(x, z, w):
					
	
	
	def run(self):
		#for now iterating one round
		#number of samples
		
		for i in xrange(self.n):
			
			
			
			
			
			
			
			x = self.observed[i]
			
			#for now only one sample
			mu2, sig2 = self.SEP_factor
			mu2, sig2 = util.exponent_diag_gaugg(mu2, sig2, self.n)
			mu1, sig1 = self.SEP_prior
			qmu, qsig = util.product_diag_gauss(mu1, mu2, sig1, sig2)
			
			theta = np.random.multivariate_normal(qmu, np.diag(qsig))
			W = theta.reshape((3,2))
			
			zmu = exact.get_mu(self, x, W)
			zsig = exact.get_sigma(self, W, self.sigx)
			
			

			
			temp = lambda x: util. 
			row = lambda u, V, x: 
			
			
			#update SEP factors

if __name__ == "__main__":
	fa = Fa(10, 3, 2)
	fa.run()





