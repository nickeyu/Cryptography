import tensorflow as tf

def Backward_Propagation(inputs, outputs, N, width):
    # Define parameters
    learning_rate = 0.004
    num_epochs = 3000

    hl1_neurons = 60;
    
    # Build the graph
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [width])

    W1 = tf.Variable(
        tf.random_normal(shape=[width, hl1_neurons],
                        mean=0.5, stddev=1.0/np.sqrt(width), dtype=tf.float32
                        ),
        dtype=tf.float32
        )

    B1 = tf.Variable(
        tf.zeros(shape=[hl1_neurons], stddev=1/(np.sqrt(width)), dtype=tf.float32)
        )

    W_out = tf.Variable(
           tf.random_normal(shape=[width, hl1_neurons],
                        mean=0.5, stddev=1.0/np.sqrt(width), dtype=tf.float32
                        ),
        dtype=tf.float32
        )

    B_out = tf.Variable(
           tf.zeros(shape=[width], stddev=1/(np.sqrt(width)), dtype=tf.float32)
        )


    # Without activation function
    # Z = tf.reduce_sum(W * A + B)
    # With activation function
    Z1 = tf.reduce_sum(tf.tanh(W1 * X + B1))
    
    Z2 = tf.reduce_sum(tf.tanh(W_out * Z1) + B_out)

    init = tf.global_variables_initializer()

    # Loss & Optimizer
    Y = tf.placeholder(dtype=tf.float32)
    loss = tf.losses.mean_squared_error(Y, Z2)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)

    # Define and run the session
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            total_loss = 0
            print("\nEpoch {}/{}".format(epoch + 1, num_epochs))
            for i in range(N):
                _, out, out_loss = sess.run(
                    [optimizer, Z2, loss],
                    feed_dict={X: inputs[i], Y: outputs[i]}
                    )
                total_loss += out_loss
                print("Entry {}: {:0.4f} (Expected: {:0.4f})".format(i, out, outputs[i]))
        #      print(out)
        #      print(outputs[i])
            weight_hidden = sess.run(W1)
            weight_out = sess.run(W_out)
            labels = sess.run(B1) 
            labels_out = sess.run(B_out)
            print("Total loss: {:0.4f}".format(total_loss))
        print(weight_hidden)
        print(labels)
        print(weight_out)
        print(labels_out)

    return weights, labels

