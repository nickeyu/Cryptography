import tensorflow as tf
import numpy as np

def Forward_Propagation(inputs, N, width, W1, B1, W2, B2, random_mapping):
  hl1_neurons = 26
  # Build the graph
  tf.reset_default_graph()
  X = tf.placeholder(dtype=tf.float32)

  W_1 = W1
  B_1 = B1
  W_2 = W2
  B_2 = B2

  if(random_mapping):
    W_1 = tf.Variable(
                      tf.random_normal(shape=[width, hl1_neurons],
                                       mean=0, stddev=1.0, dtype=tf.float32
                                      )
                     )

    B_1 = tf.Variable(
                      tf.random_normal(shape=[hl1_neurons], 
                                       dtype=tf.float32
                                      )
                     )

    W_2 = tf.Variable(    #60 by 5 
                      tf.random_normal(shape=[hl1_neurons, 1],
                                       mean=0, stddev=1.0, dtype=tf.float32
                                      )
                     )

    B_2 = tf.Variable(
                      tf.random_normal(shape=[1], 
                                       dtype=tf.float32)
                     )

  # Without activation function
  # Z = tf.reduce_sum(W * A + B)
  # With activation function
  Z1 = tf.add(tf.matmul(X, W_1), B_1)
  Z1 = tf.sigmoid(Z1)
  Z2 = tf.add(tf.matmul(Z1, W_2), B_2)
  Z2 = tf.sigmoid(Z2)

  init = tf.global_variables_initializer()
  output = []
  # Define and run the session
  with tf.Session() as sess:
    sess.run(init)
    for i in range(N):
      out = sess.run(Z2, feed_dict={X: np.array(inputs[i])[np.newaxis]})
#      out = sess.run(Z2, feed_dict={X: np.array(inputs[i])[np.newaxis]})
      #print("Entry {}: {:0.4f}".format(i, out))
      output.append(out)
 
  return output

