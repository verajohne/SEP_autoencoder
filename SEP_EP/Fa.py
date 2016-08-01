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
		self.V = tf.Variable(np.zeros([f,f]))

		#latent variables -  mean and precision
		self.R = tf.Variable(np.zeros([dimz, dimx]))
		self.S = tf.constant(np.identity(dimz))


	def get_q(self):
		n = tf.constant(self.n, dtype = tf.float64)
		prec = tf.mul(self.V, n)
		
		prec = tf.add(prec, self.SEP_prior_prec)
		qsig = tf.matrix_inverse(prec)

		mu = tf.matmul(self.V, self.u)
		mu = tf.mul(n, mu)
		qmu = tf.matmul(qsig, mu)

		return qmu, qsig


	def sample_theta(self):
		#TODO change to precision
		#n = tf.constant(self.n , dtype = tf.float64)
		#mu, sig = util.tf_exponent_gauss(self.u, self.V, n)
		#prior cov is identity- precision and covariance are identical
		#qmu, qsig = util.tf_prod_gauss(self.SEP_prior_mean, mu, self.SEP_prior_prec, sig)

		qmu, qsig = self.get_q()
		
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
		z = tf.reshape(z, [self.dimz,1])

		mu = tf.matmul(w,z)
		mu = tf.reshape(mu, shape = [self.dimx,])
		px = tf.contrib.distributions.MultivariateNormal(mu, np.identity(self.dimx))
		px = px.pdf(tf.reshape(x, [self.dimx,]))
		return tf.mul(pz,px)

	def gamma(self, x, z, w):
		num = self.evaluate_joint(x,z,w)

		#SEP part
		f = self.dimx*self.dimz
		u = tf.reshape(self.u, shape = [f,1])
		theta = tf.reshape(w, shape = [f,1])
		pq = util.tf_evaluate_gauss(u, self.V, theta)

		#recogniton model part
		mu = tf.matmul(self.R,x)	#TODO add noise
		mu = tf.reshape(mu, shape = [self.dimz,])
		f = tf.contrib.distributions.MultivariateNormal(mu, self.S)
		f = f.pdf(tf.reshape(z, shape = [self.dimz,]))


		den = tf.mul(pq, f)

		return tf.div(num, den)

	def objective(self, x):
		#TODO:
		#modify to specify number of samples
		z = self.sample_z(x, 1)
		w = self.sample_theta()
	
		gamma = self.gamma(x,z,w)

		return tf.mul(gamma, tf.log(gamma))

	def get_log_likelihood(self):
		pass

	def fit(self, sess, n_iter, learning_rate):
		

		x_ph = tf.placeholder(tf.float64, shape = [self.dimx,1])

		objective_function = self.objective(x_ph)

		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(objective_function)


		init = tf.initialize_all_variables()
		sess.run(init)

		obj_values = []

		for j in xrange(n_iter):
			for i in xrange(self.n):
				x = self.observed[i].reshape((self.dimx,1))
			
				_, R, u, V, of = sess.run((optimizer, self.R, self.u, self.V, objective_function), feed_dict = {x_ph:x} )
				
				if i % self.n == 0:
					obj_values.append(of)
					#print of
					ll = self.get_log_likelihood()

		np.save('obj', of)
		'''
		TODO:
		check that parameters are updated
		add convergence test
		'''





