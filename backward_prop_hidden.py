import tensorflow as tf
import numpy as np

def Backward_Propagation(inputs, outputs, N, width, W_1, B_1, W_2, B_2, prev_weights_bool):
    # Define parameters
    learning_rate = 1
    num_epochs = 100

    hl1_neurons = 26
    total_loss = 0

    # Build the graph
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, width])

    W1 = tf.Variable(W_1)
    B1 = tf.Variable(B_1)
    W_out = tf.Variable(W_2)
    B_out = tf.Variable(B_2)

    if(prev_weights_bool == 0):
        W1 = tf.Variable(
            tf.random_normal(shape=[width, hl1_neurons],
                            mean=0, stddev=1.0, dtype=tf.float32
                            )
            )

        B1 = tf.Variable(
            tf.random_normal(shape=[hl1_neurons], 
                            dtype=tf.float32
                            )
            )

        W_out = tf.Variable(    #60 by 5 
               tf.random_normal(shape=[hl1_neurons, 1],
                            mean=0, stddev=1.0, dtype=tf.float32
                            )
            )

        B_out = tf.Variable(
               tf.random_normal(shape=[1], 
                            dtype=tf.float32)
            )


    #keep_prob = tf.placeholder(tf.float32)
    # Without activation function
    # Z = tf.reduce_sum(W * A + B)
    # With activation function
    #Z1 = tf.reduce_sum(tf.tanh(tf.matmul(X, W1) + B1))
    Z1 = tf.add(tf.matmul(X,W1), B1)
#    Z1 = tf.nn.relu(Z1)
    Z1 = tf.sigmoid(Z1)
#    Z1 = tf.reduce_sum(tf.tanh(tf.add(tf.matmul(X,W1), B1)))
    #Z1 = tf.nn.dropout(Z1, keep_prob)
    #Z2 = tf.reduce_sum(tf.tanh(tf.matmul(Z1, W_out) + B_out))
#    Z2 = tf.reduce_sum(tf.tanh(Z1 * W_out + B_out))
#    Z2 = tf.nn.softmax(tf.add(tf.matmul(Z1, W_out), B_out))

    Z2 = tf.add(tf.matmul(Z1, W_out), B_out)
    Z2 = tf.sigmoid(Z2)
    
    initial = tf.global_variables_initializer()

    # Loss & Optimizer
    Y = tf.placeholder(dtype=tf.float32)
    loss = tf.losses.mean_squared_error(Y, Z2)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)

    # Define and run the session
    with tf.Session() as sess:
        sess.run(initial)
        for epoch in range(num_epochs):
            total_loss = 0
            if((epoch + 1) % 100 == 0):
                print("\nEpoch {}/{}".format(epoch + 1, num_epochs))
            for i in range(N):
                #i = np.array(inputs[i])[np.newaxis]
                #np.expand_dims(X, axis=0)
                _, out, out_loss = sess.run(
                    [optimizer, Z2, loss],
                    feed_dict={X: np.array(inputs[i])[np.newaxis], Y: outputs[i]}
                    )
                total_loss += out_loss
                #print("Entry: %s" % out)
                #print("Entry {}: {:0.4f} (Expected: {:0.4f})".format(i, out, outputs[i]))
        #      print(out)
        #      print(outputs[i])
            weight_hidden = sess.run(W1)
            weight_out = sess.run(W_out)
            labels = sess.run(B1) 
            labels_out = sess.run(B_out)
            if((epoch + 1) % 100 == 0):
                print("Total loss: {:0.4f}".format(total_loss))
#        print(weight_hidden)
#        print(labels)
#        print(weight_out)
#        print(labels_out)

    return weight_hidden, labels, weight_out, labels_out, total_loss

