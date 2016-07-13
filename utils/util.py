
import numpy as np

def generate_data(n, W, sigx, dimx, dimz):
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