import sys
sys.path.append('../')

from utils import util
from exact import exact

import numpy as np

class Fa(object, n, dimx, dimz):
	
	def __init__(self,):
	
		self.n = n
		self.W = np.random.normal(0,1, size = (dimx,dimz))
		self.sigx = 0.0001
		
		#data
		data = util.generate_data(n, self.W, self.sigx, dimx, dimz)
		self.observed = data[0]
		self.latent = data[1]
		
		#SEP params: (means, exponent values of covariance matrix (e^-b) - b on the diagonal)
		f = dimx*dimz
		self.SEP_prior = (np.zeros(), np.zeros(f))
		self.SEP_factor = (np.zeros(), np.array([-10**4]*f))
		
		
	
		