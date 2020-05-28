import numpy as np

def checkSize(w, X, y):
	# w: 785 by 10 matrix
	# X: N by 785 matrix
	# y: N by 1 matrix
	assert y.dtype == 'int'
	assert X.shape[0] == y.shape[0]
	assert X.shape[1] == w.shape[0]
	assert len(y.shape) == 2
	assert y.shape[1] == 1

def loss(w, X, y):
	"""
	Optional
	Useful to run gradient checking
	Utilize softmax function below
	"""
	checkSize(w, X, y)

def grad(w, X, y):
	"""
	Return gradient of multiclass softmax
	Utilize softmax function below
	"""
	checkSize(w, X, y)
	a = np.array([np.argmax(np.dot(X,w),axis=1)]).T
	b = np.dot(X,w)
	i = 0
	for i in len(y)
		b[i,a[i,0]] = b[i,a[i,0]]-1
		i += 1
	grad = np.dot(X.T,(softmax(w,X)-b))
	
	return grad


def softmax(w, X):
	scores = np.matmul(X, w)
	maxscores = scores.max(axis = 1)
	scores = scores - maxscores[:, np.newaxis]
	exp_scores = np.exp(scores)

	sum_scores = np.sum(exp_scores, axis = 1)
	return exp_scores/sum_scores[:, np.newaxis]

def predict(w, X):
	"""
	Prediction
	"""
	X = X.T
	ylab = np.array([np.argmax(np.dot(X.T,w),axis=1)]).T
	I = (y==ylab)
	
	return I

def accuracy(w, X, y):
	"""
	Accuracy of the model
	"""
	accu = 1 - (1/len(y))*(predict(w,X).sum())
	
	return accu