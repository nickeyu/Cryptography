import tensorflow as tf

def Forward_Propagation(inputs, N, width, W, B, random_mapping):
  # Build the graph
  tf.reset_default_graph()
  X = tf.placeholder(dtype=tf.float32)

  if(random_mapping):
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
  output = []
  # Define and run the session
  with tf.Session() as sess:
    sess.run(init)
    for i in range(N):
      out = sess.run(Z, feed_dict={X: inputs[i]})
      print("Entry {}: {:0.4f}".format(i, out))
      output.append(out)
 
  return output

