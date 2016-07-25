import numpy as np


def get_mu(self, x, W):
	temp = np.dot(W.transpose(), W)
	temp = np.linalg.inv(temp)
	temp = np.dot(temp, W.transpose())
	return np.dot(temp, x)

def get_sigma(self, W, sigx2):
	temp = np.dot(W.transpose(), W)
	return np.linalg.inv(temp/sigx2)
	