# Abstract class

class Model(object):
	def __init__(self):
		pass

	# X is a list of text strings and y is a corresponding list
	# of 0 or 1 labels
	def fit(self, X, y):
		pass

	# Run prediction on a single text string x, return some value between
	# 0 and 1 representing the model's confidence that x is a positive value
	def predict(self, item):
		pass
