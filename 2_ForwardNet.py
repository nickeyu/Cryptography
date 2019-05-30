import tensorflow as tf

# Define parameters
N = 6  # Number of data
width = 4  # Data width
inputs = [[0, 0, 0, 1], 
          [0, 0, 1, 0],
          [0, 1, 0, 0],
          [1, 0, 0, 0],
          [1, 0, 1, 0],
          [1, 0, 1, 1]]

outputs = [0.5,
           0.3,
           0.1,
           0.8,
           0.3,
           -0.2]

# Build the graph
tf.reset_default_graph()
X = tf.placeholder(dtype=tf.float32)

W = tf.Variable(
    tf.random_normal(shape=[width, width],
                     mean=0.5,
                     stddev=1.0,
                     dtype=tf.float32
                     ),
    dtype=tf.float32
    )

B = tf.Variable(
    tf.zeros(shape=[width, 1],
             dtype=tf.float32
             ),
    )

# Without activation function
# Z = tf.reduce_sum(W * A + B)
# With activation function
Z = tf.reduce_sum(tf.tanh(W * X + B))

init = tf.global_variables_initializer()

# Define and run the session
with tf.Session() as sess:
  sess.run(init)
  for i in range(N):
    out = sess.run(Z, feed_dict={X: inputs[i]})
    print("Entry {}: {:0.4f}".format(i, out))
  