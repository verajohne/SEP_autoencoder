import autograd.numpy as np
from autograd import grad 
from numpy.linalg import inv

def generate_data(n, W, sigx, dimx, dimz):
	#factor analysis
	observed = np.zeros([n, dimx])
	latent = np.zeros([n, dimz])
	
	for i in xrange(n):
		#latent variable
		z = np.random.normal(0,1, size = (dimz,))

		#observed
		mu = np.dot(W,z)
		cov = sigx*np.identity(dimx)
		x = np.random.multivariate_normal(mu, cov)
		
		observed[i] = x
		latent[i] = z

	return observed, latent
		
def gradient_descent(objFunc, w):
	dfunc = grad(objFunc)
	lrate = 0.00001
	a = []
	for i in range(5000):
		a.append(objFunc(w))
		improv = lrate * dfunc(w)
		w = w - improv
		if i % 1000 == 0:
			lrate = lrate/10.0
	return w,a

def product_gaussians(mu1, mu2, sig1, dig2):
	assert mu1.shape == mu2.shape, "Mu dimensions are not the same"
	assert sig1.shape == sig2.shape, "Sigma dimensions are not the same"
	prec1 = inv(sig1)
	prec2 = inv(sig2)
	
	sig = np.lingalg.inv( prec1 + prec2 )
	temp = np.dot(prec1, mu1) + np.dot(prec2, mu2)
	mu = np.dot(sig, temp)
	return mu, sig
	
	
	
	
	