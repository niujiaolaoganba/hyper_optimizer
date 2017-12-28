# -*- coding: utf-8 -*-
"""
@authon: Charles Liu <guobinliulgb@gmail.com>
@brief: splitter data
@todo:  
** feature_combiner **  

"""

import numpy as np 
import pandas as pd 
import sklearn.cross_validation
from sklearn.cross_validation import ShuffleSplit, _validate_shuffle_split
import matplotlib.pyplot as plt 
from matplotlib_venn import venn2
plt.rcParams["figure.figsize"] = [5, 5]

import config
from utils import pkl_utils

class StratifiedShuffleSplit(sklearn.cross_validation.StratifiedShuffleSplit):
	"""docstring for StratifiedShuffleSplit"""
	def __init__(self, y, n_iter=10, test_size=0.1, train_size=None, random_state=None):
		n = len(y)
		self.y = np.array(y)
		self.classes, self.y_indices = np.unique(y, return_inverse=True)
		self.random_state = random_state
		self.train_size = train_size
		self.test_size = test_size
		self.n_iter = n_iter
		self.n = n
		self.n_train, self.n_test = _validate_shuffle_split(n, test_size, train_size)

class HomedepotSplitter:
	def __init__(self, dfTrain, dfTest, n_iter=5, random_state=config.RANDOM_SEED,
				verbose=False, plot=False, split_param=[0.5,0.25,0.5]):
	self.dfTrain = dfTrain
	self.dfTest = dfTest
	self.n_iter = n_iter
	self.random_state = random_state
	self.verbose = verbose
	self.plot = plot
	self.split_param = split_param

	def __str__(self):
		return "HomedepotSplitter"

	def _check_split(self, dfTrain, dfTest, col, suffix="", plot=""):
		if self.verbose:
			print("-"*50)
		num_train = dfTrain.shape[0]
		num_test = dfTest.shpae[0]
		ratio_train = num_train/(num_train + num_test)
		ratio_test = num_test/(num_test + num_train)
		if self.verbose:
			print("Sample State: %.2f (train) | %.2f (test)" % (ratio_train, ratio_test))

		puid_train = set(np.unique(dfTrain[col]))
		puid_test = set(np.unique(dfTest[col]))
		puid_total = puid_train.union(puid_test)
		puid_intersect = puid_train.intersection(puid_test)

		ratio_train = ((len(puid_train) - len(puid_intersect)) / len(puid_total))
		ratio_intersect = len(puid_intersect) / len(puid_total)
		ratio_test = ((len(puid_test) - len(puid_intersect)) / len(puid_total))

		if self.verbose:
			print("%s States: %.2f (train) | %.2f (train & test) | %.2f (test)" % (col, ratio_train, ratio_intersect, ratio_test))

		if (plot == "" and self.plot) or plot:
			plt.figure()
			if suffix == "actual":
				venn2([puid_train, puid_test], ("train", "test"))
			else:
				venn2([puid_train, puid_test], ("train", "valid"))
			fig_file = "%s/%s_%s.pdf" % (config.FIG_DIR, suffix, col)
			plt.savefig(fig_file)
			plt.clf()

		puid_train = sorted(list(puid_train))
		return puid_train

	def split(self):
		if self.verbose:
			print("*"*50)
			print("Naive split")










		