import tensorflow as tf
import numpy as np

from utils import util

from SEP_EP.Fa import Fa

def main(n_iter = 10, learning_rate = 1e-50, num_samples = 1):

	sess = tf.Session()

	data_size = 10
	model = Fa(data_size, dimx =3, dimz = 2)
	model.fit(sess, n_iter = n_iter, learning_rate = learning_rate)










if __name__ == "__main__":
	main()