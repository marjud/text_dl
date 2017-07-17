import tensorflow as tf
from deep_learning_text import create_feature_sets_labels
import numpy as np
#imput data from deep_learning_text,py
train_x, train_y, test_x, test_y = create_feature_sets_labels('pos.txt', 'neg.txt')

n_node_hl1 = 500
n_node_hl2 = 500
n_node_hl3 = 500

n_classes = 2
batch_size = 100 # go through 100 batches features at a time

#squash 28*28 into 784 features
#placeholder is a promise to value later
x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')


def neural_network_model(data):
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_node_hl1])),
						'biases': tf.Variable(tf.random_normal([n_node_hl1]))}
	
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_node_hl1, n_node_hl2])),
					   'biases': tf.Variable(tf.random_normal([n_node_hl2]))}
	
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_node_hl2, n_node_hl3])),
					   'biases': tf.Variable(tf.random_normal([n_node_hl3]))}
	
	output_layer = {'weights':tf.Variable(tf.random_normal([n_node_hl3, n_classes])),
					   'biases': tf.Variable(tf.random_normal([n_classes]))}
	#what happens at each layer, add and matmul.
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	#activation function (rectified linear)
	l1 = tf.nn.relu(l1)
	
	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)
	
	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)
	
	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
	return output
#computation graph/ model is complete. now need to tell tensorflow what to do

saver = tf.train(saver)
tf_log = 'tf_log'
def train_neural_network(x):
	#the output from the above def is a onehot array
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	#minimize cost
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	#cycles of feedforward + backprop
	hm_epochs = 10
	
	#training the network
	#use with becuase it automatically closes session
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range (hm_epochs):
			epoch_loss = 0
			
			i = 0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x= np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				
				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c
				i += batch_size
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
		
		
		#once the weights are optimized, run through model	
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:test_x, y: test_y}))

train_neural_network(x)