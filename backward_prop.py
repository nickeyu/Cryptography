import tensorflow as tf

def Backward_Propagation(inputs, outputs, N, width):
    # Define parameters
    learning_rate = 0.004
    num_epochs = 700

    # Build the graph
    tf.reset_default_graph()
    X = tf.placeholder(dtype=tf.float32)

    W = tf.Variable(
        tf.random_normal(shape=[width, width],
                        mean=0.5, stddev=1.0, dtype=tf.float32
                        ),
        dtype=tf.float32
        )

    B = tf.Variable(
        tf.zeros(shape=[width, 1], dtype=tf.float32),
        )

    # Without activation function
    # Z = tf.reduce_sum(W * A + B)
    # With activation function
    Z = tf.reduce_sum(tf.tanh(W * X + B))

    init = tf.global_variables_initializer()

    # Loss & Optimizer
    Y = tf.placeholder(dtype=tf.float32)
    loss = tf.losses.mean_squared_error(Y, Z)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)

    # Define and run the session
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            total_loss = 0
            #print("\nEpoch {}/{}".format(epoch + 1, num_epochs))
            for i in range(N):
                _, out, out_loss = sess.run(
                    [optimizer, Z, loss],
                    feed_dict={X: inputs[i], Y: outputs[i]}
                    )
                total_loss += out_loss
                #print("Entry {}: {:0.4f} (Expected: {:0.4f})".format(i, out, outputs[i]))
        #      print(out)
        #      print(outputs[i])
            weights = sess.run(W)
            biases = sess.run(B) 
            #print("Total loss: {:0.4f}".format(total_loss))

    return weights, biases

