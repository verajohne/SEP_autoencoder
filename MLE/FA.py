import sys
sys.path.append('../')

from utils import util
from utils import plotter
import matplotlib.pyplot as plt

#import numpy as np
import autograd.numpy as np
import scipy as sp
from scipy.optimize import minimize


from numpy.random import RandomState
import pandas as pd
from autograd import grad 

RS = RandomState(1213)

class FA(object):
	
	def __init__(self,n, dimz = 2, dimx = 3):
		
		self.n = n
		self.sigx = 0.000001
		#sigw = 1#RS.normal(0,1)
		self.W = self.W = RS.normal(0,1, size = (dimx,dimz))
		self.dimz = dimz
		self.dimx = dimx
		
		data = util.generate_data(n, self.W, self.sigx, dimx, dimz)
		self.observed = data[0]
		self.latent = data[1]
	
	def get_mu(self, x, W):
		temp = np.dot(W.transpose(), W)
		temp = np.linalg.inv(temp)
		temp = np.dot(temp, W.transpose())
		return np.dot(temp, x)
		
		
	def marginal_likelihood(self, W0):
		a = self.sigx*np.identity(self.dimx)
				
		win = lambda w: np.dot(w, w.transpose()) + a
		const = lambda w: -(self.n/2.0)*np.log( np.linalg.det(win(w)) )
		
		pdin = lambda w: np.linalg.inv( win(w) )
		
		pd = lambda w,i: np.dot(np.dot(self.observed[i].transpose(), pdin(w)), self.observed[i])
		
		final = lambda w: sum(pd(w, i)  for i in range(self.n))
		
		evidence = lambda w: - const(w) + 0.5*final(w)
		gradient = grad(evidence)

		ans, a = util.gradient_descent(evidence, W0)
		plt.plot(a)
		print a[0], a[-1]
		plt.show()
		
		#print ans
		return ans
		
	def MLE_EP(self, random_init):
		w_init = RS.normal(0,1, (self.dimx, self.dimz))
		if random_init is False:
			print "FALSE"
			w_init = self.W
		mus = np.array([])
		#w = self.marginal_likelihood(w_init)
		print "initialisation of W:"
		print w_init
		print ""
		w = self.marginal_likelihood(w_init)
		print "True W"
		print self.W
		print "MLE W"
		print w
		
		mus = np.array([])
		
		for i in xrange(self.n):
			mu = self.get_mu(self.observed[i], w)
			mus = np.hstack((mus, mu))
		mus = mus.reshape((self.n,2))
		sig = np.dot(self.W.transpose(), self.W)
		sig = sig/self.sigx
		sig = np.linalg.inv(sig)

		return mus, sig
		
		
		


if __name__ == "__main__":
	fa = FA(10)
	w = RS.normal(0,1, (3, 2))
	mus, sig = fa.MLE_EP()
	#print fa.W
	#plotter.plot_approx_posterior(sig, mus, 0)
	#print "Truth: ", fa.W
	#print ""
	#print "init: ", w
	#fa.marginal_likelihood(w)
	#print ""
	#print "init: ", fa.W
	#print fa.W
	#fa.marginal_likelihood(fa.W)
	#fa.marginal_likelihood(w)
		










	