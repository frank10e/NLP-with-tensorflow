# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# tensorflow import
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Simple TensorFlow Example
hello = tf.constant("Hello Tensorflow!")
sess = tf.Session()
print(sess.run(hello))
sess.close()


# Calculate a and b
a = tf.constant(3.0, dtype = tf.float32)
b = tf.constant(4.0, dtype = tf.float32)

sum_a_b = tf.add(a,b)
sess = tf.Session()
print(sess.run(sum_a_b))
sess.close()
     

# Running a Graph using a Tensorflow Session
# Configure
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config = config)

with tf.Session(config=config) as sess:
    print(sess.run(sum_a_b))

with tf.Session(config=config) as sess:
    first_const,sum_result = sess.run([a, sum_a_b])
    print("The first constant tensor has value: {}".format(first_const))
    print("The resukt of the add operation has value: {}".format(sum_result))

# Placeholders

x = tf.constant([[1.0],[2.0]], dtype = tf.float32)
W = tf.constant([[3.0,4.0],[5.0,6.0]], dtype=tf.float32)
y = tf.matmul(W,x) # perform matrix-vector multiplication W*x

with tf.Session() as sess:
    print(sess.run(y))

x = tf.placeholder(tf.float32, shape=[2,1])
W = tf.constant([[3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
y = tf.matmul(W, x)

# TensorFlow: feed_dict (multiple values OR numpy vs array OR dataset OR 
# tensor OR list OR batch)

# TensorFlow session run without feed_dict
# TensorFlow eval feed_dict
# TensorFlow cannot interpret the feed_dict key as tensor
# Link: https://pythonguides.com/tensorflow-feed_dict/
with tf.Session() as sess:
    print("x is [[1.0], [2.0]]:")
    print(sess.run(y, feed_dict={x: [[1.0], [2.0]]})) # input a feed_dict for placeholder x -- must be at least rank-2!
    print("x is [[2.0], [4.0]]:")
    print(sess.run(y, feed_dict={x: [[2.0], [4.0]]})) # we can change input to graph from here

x = tf.placeholder(tf.float32, shape=[2,1])
init_value = tf.random_normal(shape = [2, 2]) # will draw a 2 x 2 matrix with entries from a standard normal distn
W = tf.Variable(init_value) # Within the graph, initialize W with the values drawn from a standard normal above
y = tf.matmul(W, x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # necessary step now that we have variables
    print("Our random matrix W:\n")
    print(sess.run(W)) # Notice that we don't have to use a feed_dict here, because x is not part of computing W
    print("\nResult of our matrix multiplication, y:\n")
    print(sess.run(y, feed_dict={x: [[1.0], [2.0]]}))

x = tf.placeholder(tf.float32)
W = tf.get_variable(name="W", shape = [2, 2], initializer=tf.random_normal_initializer) # note we give the variable a name
y = tf.matmul(W, x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(W))
    print(sess.run(y, feed_dict={x: [[1.0], [2.0]]}))
    
    