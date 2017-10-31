import csv
import numpy
from sklearn import datasets

def load_file(filename):

	raw_data = open(filename, 'rt')
	reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
	data = numpy.array(list(reader))
	X = data[:,:len(data[0]) - 1]
	X = numpy.array(X).astype('float')
	Y = data[:, -1]
	return (X,Y)

def load_iris():
	iris = datasets.load_iris()
	return (iris.data, iris.target)

def load_bank_auth():
	X,Y =  load_file('datasets/data_ba.txt')
	Y = numpy.array(Y).astype('int')
	return (X,Y)

def load_ionosphere():
	X, Y = load_file('datasets/ionosphere.data')


	for i in range(len(Y)):
		if Y[i] == 'g':
			Y[i] = 1
		elif Y[i] == 'b':
			Y[i] = 0

	return (X, Y)

def load_magic():
	X, Y = load_file('datasets/magic04.data')

	for i in range(len(Y)):
		if Y[i] == 'g':
			Y[i] = 0
		elif Y[i] == 'h':
			Y[i] = 1

	return (X, Y)
