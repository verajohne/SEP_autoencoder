import sys
sys.path.append('../')

from utils import util
from exact import exact
import gaussian

import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as mvn


class Fa(object):
	
	def __init__(self, n, dimx, dimz):
	
		self.n = n
		self.sigx = 0.0001
		#sigw = np.random.normal(0,1)
		self.W = np.random.normal(0,1, size = (dimx,dimz))
		self.dimz = dimz
		self.dimx = dimx
		#data
		data = util.generate_data(n, self.W, self.sigx, dimx, dimz)
		self.observed = data[0]
		self.latent = data[1]
		
		#SEP params - mean and precision
		f = dimx*dimz
		self.SEP_prior_mean = tf.constant(np.zeros(f).reshape((f,1)))
		self.SEP_prior_prec = tf.constant(np.identity(f))

		self.u = tf.Variable(np.zeros(f).reshape((f,1)))
		#W = tf.reshape(theta, shape = [self.dimx, self.dimz])
		#self.V = tf.Variable( np.diag(np.array([1e-6]*f)) )
		self.V = tf.Variable( np.diag(np.array([1.0]*f)) )
		#self.V = tf.Variable(np.zeros([f,f]))

		#latent variables -  mean and precision
		self.R = tf.Variable(np.zeros([dimz, dimx]))
		self.S = tf.constant(np.identity(dimz))

	def sample_theta(self):
		#TODO change to precision
		n = tf.constant(self.n - 1, dtype = tf.float64)
		mu, sig = util.tf_exponent_gauss(self.u, self.V, n)
		#prior cov is identity- precision and covariance are identical
		qmu, qsig = util.tf_prod_gauss(self.SEP_prior_mean, mu, self.SEP_prior_prec, sig)
		
		qmu = tf.reshape(qmu, shape = [6,])
		q = tf.contrib.distributions.MultivariateNormal(qmu, qsig)
		theta = q.sample(1)

		W = tf.reshape(theta, shape = [self.dimx, self.dimz])
		return W		
		

	def sample_z(self, x, num):
		mu = tf.matmul(self.R, x)
		g = gaussian.Gaussian_full(self.dimz, mu, self.S)
		return g.sample(1)
	
	def evaluate_joint(self, x, z, w):
		#z = z.eval()
		pz = tf.contrib.distributions.MultivariateNormal(np.zeros(self.dimz), np.identity(self.dimz))
		pz = pz.pdf(tf.reshape(z, [self.dimz,]))
		z = tf.reshape(z, [2,1])

		mu = tf.matmul(w,z)
		mu = tf.reshape(mu, shape = [self.dimx,])
		px = tf.contrib.distributions.MultivariateNormal(mu, np.identity(self.dimx))
		px = px.pdf(tf.reshape(x, [self.dimx,]))
		return tf.mul(pz,px)

	def gamma(self, x, z, w):
		num = self.evaluate_joint(x,z,w)

		V_cov = tf.matrix_inverse(self.V) #Will this cause a problem -- inversion of 0
		u = tf.reshape(self.u, shape = [6,])
		q = tf.contrib.distributions.MultivariateNormal(u, V_cov)
		theta = tf.reshape(w, shape = [6,])
		pq = q.pdf(theta) ####CHECK THIS

		mu = tf.matmul(self.R,x)

		mu = tf.reshape(mu, shape = [self.dimz,])
		f = tf.contrib.distributions.MultivariateNormal(mu, self.S)
		f = f.pdf(tf.reshape(z, shape = [self.dimz,]))


		den = tf.mul(pq, f)

		return tf.div(num, den)

	def objective(self, x):
		z = self.sample_z(x, 1)
		w = self.sample_theta()

		gamma = self.gamma(x,z,w)

		return tf.mul(gamma, tf.log(gamma))

	def fit(self, sess, n_iter, learning_rate):
		

		x_ph = tf.placeholder(tf.float64, shape = [self.dimx,1])

		objective_function = self.objective(x_ph)

		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(objective_function)


		init = tf.initialize_all_variables()
		sess.run(init)


		for j in xrange(n_iter):
			for i in xrange(self.n):
				x = self.observed[i].reshape((self.dimx,1))
				#x = np.ones(3).reshape((3,1))
				_, R, u, V, of = sess.run((optimizer, self.R, self.u, self.V, objective_function), feed_dict = {x_ph:x} )
				print of

		'''
		TODO:
		check that parameters are updated
		add convergence test
		'''





