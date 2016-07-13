import sys
sys.path.append('../')

from utils import util

import numpy as np
import scipy as sp
from numpy.random import RandomState
import pandas as pd


RS = RandomState(1213)

class FA(object):
	
	def __init__(self,n, dimz = 2, dimx = 3):
		
		self.n = n
		self.W = RS.normal(0,1, size = (dimx,dimz))
		self.sigx = 0.0001
		self.dimz = dimz
		self.dimx = dimx
		
		data = util.generate_data(n, self.W, self.sigx, dimx, dimz)
		self.observed = data[0]
		self.latent = data[1]
	
	def get_mu(self, x, W):
		temp = np.dot(W.transpose(), W)
		temp = np.linalg.inv(temp)
		temp = np.dot(temp, W.transpose())
	
	def marginal_likelihood(self, x):
		a = 1/(2*self.sigx)
		temp = - a*np.dot(x.transpose(), x) + a*np.dot(np.dot(x.transpose(), self.pc_norm2), x)
		return self.pc_norm1*np.exp(temp)
		
	def MLE_EP():
		w = RS.normal(0,0.001, size(self.dimx, self.dimz))
		
		for j in xrange(100):
			for i in xrange(
				mu = self.get_mu(self.observed[i], w)
				w = self.marginal_likelihood(self.observed[i])
		
		
		
		


if __name__ == "__main__":
	fa = FA(10)
	print fa.observed
		










	