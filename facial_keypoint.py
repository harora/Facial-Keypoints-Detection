import csv
import numpy as np
from PIL import Image
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from tqdm import *
import theano
import theano.tensor as T
import lasagne
import numpy as np
import cPickle
from skimage.io import imread
import random


def give_image(row):

	image = np.zeros((96,96),dtype = np.uint8)
	str = row[30].split(" ")
	# image = (np.asarray(str,dtype = np.float32))
	for i in range(96):
		for j in range(96):
			image[i][j] = str[96*j + i]
	image = image.reshape(1,1,96,96)

	
	landmarks = []
	for lm in range(30):
		landmarks.append(row[lm])
	landmarks = np.asarray(landmarks,dtype = np.float32)
	landmarks = (landmarks/96).reshape(1,30)



	return image ,landmarks


def load(id):
	with open('training.csv', 'rb') as csvfile:
		imagereader = csv.reader(csvfile)
		for idx,row in enumerate(imagereader):
			if idx == id: 
				image , landmarks = give_image(row)
	return image , landmarks

		

def create_nn():
	## Setting up layers

	patch_in = lasagne.layers.InputLayer((1,1,96,96)) #Takes 10 patches of 3 X 100 X 100 as input at once
	conv1 = lasagne.layers.Conv2DLayer(patch_in,30,(3,3),pad="same",nonlinearity=lasagne.nonlinearities.rectify)

	hidden1 = lasagne.layers.DenseLayer(conv1, num_units=100)
	output1 = lasagne.layers.DenseLayer(hidden1, num_units=30,nonlinearity=T.nnet.sigmoid)


	net_output = lasagne.layers.get_output(output1) #Output for the testing layer layer

	

	# Tasting
	true_output = T.fmatrix()
	loss = lasagne.objectives.squared_error(net_output,true_output) #Output for the training layer

	# ## Baking Recipe <Updates>
	all_params = lasagne.layers.get_all_params(output1)
	# updates = lasagne.updates.adam(loss.mean(),all_params)
	updates = lasagne.updates.nesterov_momentum(loss.mean(), all_params, learning_rate = 0.5, momentum=0.9)

	# Compile everything as functions
	train = theano.function(inputs=[patch_in.input_var,true_output],outputs = [net_output,loss],updates =  updates )
	test = theano.function(inputs=[patch_in.input_var], outputs=[net_output])

	return train,test


if __name__=="__main__":
    train,test = create_nn()
    for i in range(100):
    	image ,landmarks = load(i+1)
    	result , loss = train(image,landmarks)
    	print loss.mean()



    








