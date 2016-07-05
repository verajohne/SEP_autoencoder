import numpy as np
import scipy
from numpy.random import RandomState
from scipy.stats import norm

class MVN:

	def __init__(self, mu, prec):
		self.mu = mu
		self.prec = prec

	def add_mvn(A,B):
		prec = A.prec + B.prec
		mu = np.dot(prec , np.dot(A.prec, A.mu) + np.dot(B.prec, B.mu))
		C = MVN(mu, prec)
		return C
		
