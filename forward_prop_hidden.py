import tensorflow as tf

def Forward_Propagation(inputs, N, width, W_1, B_1, W_2, B_2, random_mapping):
  hl1_neurons = 26
  # Build the graph
  tf.reset_default_graph()
  X = tf.placeholder(dtype=tf.float32)

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
                      tf.random_normal(shape=[hl1_neurons, width],
                                       mean=0, stddev=1.0, dtype=tf.float32
                                      )
                     )

    B_2 = tf.Variable(
                      tf.random_normal(shape=[width], 
                                       dtype=tf.float32)
                     )

  # Without activation function
  # Z = tf.reduce_sum(W * A + B)
  # With activation function
  Z1 = tf.sigmoid(tf.add(tf.matmul(X, W_1), B_1))
  Z2 = tf.sigmoid(tf.add(tf.matmul(Z1, W_2), B_2))

  init = tf.global_variables_initializer()
  output = []
  # Define and run the session
  with tf.Session() as sess:
    sess.run(init)
    for i in range(N):
      out = sess.run(Z1, Z2, feed_dict={X: inputs[i]})
      #print("Entry {}: {:0.4f}".format(i, out))
      output.append(out)
 
  return output

