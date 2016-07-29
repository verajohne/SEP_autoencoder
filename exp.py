import tensorflow as tf
import numpy as np

from utils import util

from SEP_EP.Fa import Fa

def main(n_iter = 10, learning_rate = 1e-50, num_samples = 1):

	print "starting..."
	sess = tf.Session()

	model = Fa(10, 3, 2)
	model.fit(sess, n_iter = n_iter, learning_rate = learning_rate)










if __name__ == "__main__":
	main()