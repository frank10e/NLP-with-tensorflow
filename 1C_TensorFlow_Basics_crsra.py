#!/usr/bin/env python
# coding: utf-8

# # Tensorflow Introduction

# TensorFlow is Google's software library for machine learning. The TensorFlow API is quite extensive and has many functions for creating deep learning models at various levels of abstraction. To get started, however, we'll focus on some TensorFlow basics, without any deep learning. You should already be reasonably comfortable with Python before proceeding (see 0A_Python_Prerequisites.ipynb). 

# ## Why is TensorFlow Needed?

# ### Limitations of Python and NumPy

# Since we've just reviewed Python and NumPy usage, you may be wondering why we need TensorFlow for Deep Learning, instead of staying within the confines of only Python or NumPy. While certain aspects of the Python language make it convenient to use in many cases, they come with efficiency tradeoffs that render pure Python nearly unuseable when working with the large datasets necessary for training deep learning models.  For instance, Python is *dynamically typed*, which means that Python stores in memory the information about a variable's type. This requires extra memory allocation and also necessitates what are called *runtime type checks*, in which Python looks up a variable's type whenever operations are performed using that variable. NumPy mitigates some of these inefficiencies with its arrays, which are more compactly stored in memory because all elements in an array are of the same type (compared to, say, a Python list). NumPy takes advantage of this compact representation to perform quick array manipulations using operations implemented in C. 
# 
# While individual NumPy operations are efficient, stringing these operations together introduces a slowdown.  This is because between each NumPy operation, computation goes back and forth between the Python interpreter and underlying C code of NumPy. NumPy also experiences another drawback specific to deep learning. As you'll learn, a canonical algorithm for training deep models uses *backpropagation*, which requires the calculation of gradients. Before TensorFlow and other similar libraries, programmers manually (i.e., using pen and paper) did the calculus, deriving the symbolic gradient of the function to be minimized, then writing special code to take partial derivatives at an arbitrary input point. This is mechanical work that a computer should be able to do automatically. But NumPy's structure does not provide an easy way of computing these derivatives automatically. Why? Automatically computing the derivative of some formula requires having some representation of that formula in memory. But when you run NumPy operations, they simply execute and return their results; no trace is left of the steps used to get from first input to final output. There is no easy way to go back and compute derivatives later on in a program.

# ### TensorFlow Solutions

# TensorFlow addresses the above limitations of NumPy using a paradigm known as the [*computational graph*](https://www.tensorflow.org/api_docs/python/tf/Graph).  In this paradigm, a programmer describes the entire desired computation, capturing it in a representation known as a "graph."  To use graph-theory parlance, the nodes of a computation graph are the operations to be performed, and the edges of the graph represent the data flowing from one operation to another. Constructing a graph establishes the formula of the computation and stores it in memory.  
# 
# With a representation of the entire computation, TensorFlow is able to perform all operations at once using low-level implementations written in C and C++. This increases the efficiency of the computation. Additionally, TensorFlow can perform automatic differentiation based on the structure of the computation graph.

# ## Using TensorFlow

# ### Simple TensorFlow Example

# First, let's try a TensorFlow variant of the canonical "Hello, World" program: 

# In[1]:


import tensorflow as tf # When we import TensorFlow, a default graph is made

hello = tf.constant("Hello, TensorFlow!") # Add a constant-operation to the (default) graph
sess = tf.Session() # Create a session from which to run the graph
print(sess.run(hello))
sess.close()


# Congratulations! If TensorFlow is new to you, then you have just run your first TensorFlow program. Despite how short it is, there is plenty to discuss about what TensorFlow is doing in the lines above. We'll do that in the sections below.

# ### The Default Graph, Tensors, and Operations

# When we import TensorFlow, a default graph object is automatically created. A [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) object holds the computational graph that we will be defining. While TensorFlow supports the creation of multiple graphs, in practice, this is rarely done; for the rest of this course (and likely for the rest of your time using TensorFlow), you'll be using the default graph.
# 
# Upon creation, a graph is initially empty, which means no operations have been added to it. Recall that an operation is a node of the computational graph. Each operation accepts some number of inputs (0 or more) and produces 0 or more outputs, which can then possibly be passed on to other operations. The execution of an operation might also incur such side-effects as printing to the console, writing to a file, or modifying a variable in memory. The Operation object describes all of this computation, but the computation does not occur until the graph is completely built and then run.  As a shorthand for the term "TensorFlow Operation", it is not uncommon to see "op" instead.
# 
# In the second line of our "Hello, TensorFlow" example, we added a simple operation to the default graph: the `constant` operation. This op takes zero inputs and gives one output, and we tell TensorFlow what that constant output is in the `tf.constant()` function.
# 
# The inputs and outputs of a TensorFlow operation are of type `Tensor`, which is an object that is used to represent data flowing along the edges of a computational graph. Like with NumPy arrays, the data of a Tensor can only be of one type, and it can have any number of dimensions (the "rank" of the Tensor). In "Hello, TensorFlow", the single output tensor of the `tf.constant` function is `hello`. It's important to note that the string "Hello, TensorFlow" is not actually stored in the `hello` tensor. Instead, the tensor refers to that piece of data that will be computed when the graph is run. 
# 
# Let's quickly take a look at another example, this time using an operation that accepts inputs. We will again create constant operations, and now use their outputs as inputs to an addition operation.

# In[2]:


a = tf.constant(3.0, dtype=tf.float32) # add a constant-op to the graph
b = tf.constant(4.0, dtype=tf.float32) # add another constant-op to the graph
sum_a_b = tf.add(a,b) # create a TensorFlow op that adds tensors a,b and produces a new tensor
sess = tf.Session()
print(sess.run(sum_a_b))
sess.close()


# Once a node has been added to the graph, it will typically persist, which is fine (and in fact desired) for most use cases. Unfortunately, since certain ops require unique names, this behavior doesn't always play well with IPython, particularly for example when trying to demonstrate certain concepts (or re-running cells) in a Jupyter notebook. As such, we may find ourselves wanting to clear the default graph, which we do with [`tf.reset_default_graph()`](https://www.tensorflow.org/api_docs/python/tf/reset_default_graph). You may see this done in future notebooks.

# ### Running a Graph using a TensorFlow Session

# You've probably noticed that in the above examples, before printing the result of some computation, we first run something called a TensorFlow `Session`. This object interfaces between Python and the instructions for your TensorFlow graph. A `tf.Session` owns physical resources -- such as GPUs -- and so it is generally a good practice to give the Session configuration information for such things as memory allocation. This can be accomplished like so:

# In[3]:


# Configure a session to not use too much GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Create the actual session. The config argument is only necessary if you have defined a configuration, like above
sess = tf.Session(config=config)


# Another good practice, especially when working with GPUs, is to close the Session when you have completed all of your computations with it. Closing the session releases the computational resources that have been set aside by TensorFlow. In the previous examples, this was done using `sess.close()`. It is also possible to use what is called a *context-manager* by putting the session in a `with` block that closes the session when you exit the block. This looks like the following:

# In[4]:


with tf.Session(config=config) as sess:
    print(sess.run(sum_a_b))
    # the session will close when we leave the with-block


# Once you have created a session using the `tf.Session` constructor, you can use the `run` method to tell TensorFlow to actually execute the computation you have defined in the graph. When we use `sess.run()`, we pass as an argument the tensor we would like TensorFlow to compute, and TensorFlow only executes the parts of the graph necessary for that tensor. Typically, we'll want to fetch the value of multiple values, and we can do so by passing `sess.run()` a list, like so:

# In[5]:


with tf.Session(config=config) as sess:
    first_const, sum_result = sess.run([a, sum_a_b])
    print("The first constant tensor has value: {}".format(first_const))
    print("The result of the add operation has value: {}".format(sum_result))


# ### Placeholders

# Let's move away from scalars and start dealing with higher rank tensors. Here is an example illustrating a basic matrix-vector multiplication using the TensorFlow function *tf.matmul* to create the op:

# In[6]:


x = tf.constant([[1.0], [2.0]], dtype=tf.float32) # tf.matmul requires both arguments be >= rank-2
W = tf.constant([[3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
y = tf.matmul(W, x) # perform matrix-vector multiplication W * x

with tf.Session() as sess:
    print(sess.run(y))


# When we run the above graph, we will always get the same result because `x` and `W` are fixed. If we wanted to change the value of `x` in the graph, we'd have to manually change the value at the creation of the `x` tensor. TensorFlow allows us to parameterize inputs using what are called `placeholder` operations, which we then fill in later when we run the session. Placeholders are added to the graph using `tf.placeholder`, and when we do so, we also need to tell TensorFlow the type of the value that the placeholder will hold. It's also good to pass in the shape of the placeholder. We can now change our graph construction from above to look like the following:

# In[7]:


x = tf.placeholder(tf.float32, shape=[2,1])
W = tf.constant([[3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
y = tf.matmul(W, x)


# Now, when we call `sess.run`, in addition to telling TensorFlow which tensor values we'd like to fetch, we also need to pass in an argument to tell TensorFlow what to fill-an as the values for the placeholders we've put in the graph. This is accomplished using what is called a `feed_dict`, a dictionary whose keys are placeholder tensors and whose values are what to fill-in for the placeholder.

# In[8]:


with tf.Session() as sess:
    print("x is [[1.0], [2.0]]:")
    print(sess.run(y, feed_dict={x: [[1.0], [2.0]]})) # input a feed_dict for placeholder x -- must be at least rank-2!
    print("x is [[2.0], [4.0]]:")
    print(sess.run(y, feed_dict={x: [[2.0], [4.0]]})) # we can change input to graph from here


# Note from the above that we can fetch the same tensor, `y`, multiple times but with different `feed_dict` arguments for the values of tensor `x`. If we tried to fetch `y` by running the session without feeding in `x`, TensorFlow will return an error. A `feed_dict` needs to specify a value for each placeholder in the graph needed to compute the desired output. Furthermore, each call to `sess.run` wipes the placeholder values clean; if we previously fed in a value for `x` and then later tried to run the session without feeding in a value for `x`, the previous value will not have been stored anywhere and we will get an error.

# ### Variables & Initialization

# Above, we made `x` a placeholder in our graph so that we could better treat it as an input for our computation graph. However, we left `W` as a constant. Typically, a matrix like `W` is a parameter of a machine learning model, and so it is not properly considered an input, but we still need to let `W` change. In fact, learning `W` might be our goal if we don't already know it.  
# 
# Another important aspect of variables is that, unlike with placeholders, the value of a variable is persistent across runs of the graph. This means that when you use a session to run a computation graph, the value of the variable is stored in memory for the next time the graph is run using the same session. 
# 
# We can tell TensorFlow to allow `W` to change using `tf.Variable` instead of `tf.Constant`. Additionally, let's suppose that we do not know the value of matrix `W`, so instead we will randomly initialize it by randomly drawing values from a Normal distribution. We modify our previous example:

# In[9]:


x = tf.placeholder(tf.float32, shape=[2,1])
init_value = tf.random_normal(shape = [2, 2]) # will draw a 2 x 2 matrix with entries from a standard normal distn
W = tf.Variable(init_value) # Within the graph, initialize W with the values drawn from a standard normal above
y = tf.matmul(W, x)


# Now that we are using TensorFlow Variables, TensorFlow requires that we first run `tf.global_variables_initializer` because executing any parts of the graph that require variables. Notice the inclusion of running the initializer op below:

# In[10]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # necessary step now that we have variables
    print("Our random matrix W:\n")
    print(sess.run(W)) # Notice that we don't have to use a feed_dict here, because x is not part of computing W
    print("\nResult of our matrix multiplication, y:\n")
    print(sess.run(y, feed_dict={x: [[1.0], [2.0]]}))


# An alternate way to create a TensorFlow variable is to use [`tf.get_variable()`](https://www.tensorflow.org/guide/variables). As noted in the TensorFlow documentation, this is the best way to create a variable. This function fetches a variable by name (important if we share weights, which can be fairly common in certain types of models), or creates a new one if it doesn't exist. The usage is only slightly different:

# In[11]:


x = tf.placeholder(tf.float32)
W = tf.get_variable(name="W", shape = [2, 2], initializer=tf.random_normal_initializer) # note we give the variable a name
y = tf.matmul(W, x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(W))
    print(sess.run(y, feed_dict={x: [[1.0], [2.0]]}))

