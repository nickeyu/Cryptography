import tensorflow as tf
import random
import binascii

# Define parameters
learning_rate = 0.004
num_epochs = 1000
N = 95  # Number of data
width = 7  # Data width

inputs = []
outputs = []

def print_mapping(input_array):
  for i in range(len(input_array)):
    print("{} --> {} ".format(input_array[i], i))


def string_to_vect(input_string):
  temp_array = []
  for i in range(len(input_string)):
    temp_array.append(input_string[i])
  return temp_array

def vect_to_string(input_array):
  temp_string = ""
  for i in range(len(input_array)):
    temp_string += (str(input_array[i]))
  return temp_string

def generate_dictionaries(input_array):
  dec_to_vect = {}
  bin_to_dec = {}
  for i in range(len(input_array)):
    dec_to_vect[i] = input_array[i]
    bin_to_dec[vect_to_string(input_array[i])] = i
  return (dec_to_vect, bin_to_dec)

def generate_input_output():
  inputs = []
  outputs = []
  temp = []
  for i in range(N):
    outputs.append(float(i + 1) / float(N))
    binary = '{0:b}'.format(i)
    for j in range(width):
      if j < width - (len(binary)):
        temp.append(0)
      else:
        temp.append(int(binary[-(width - j)]))
    inputs.append(temp)
    temp = []
  return (inputs, outputs)    

def flip_mapping(input_array, output_array, dec_to_vect, bin_to_dec):
  inputs_flipped = [[] for i in range(N)]
  outputs_flipped = output_array
  for i in range(N):
    inputs_flipped[bin_to_dec[vect_to_string(input_array[i])]] = (dec_to_vect[i])
  return (inputs_flipped, outputs_flipped)

def print_options():
  print("Select option. Type 0 for option 0. Type 1 for option 1. Type 2 for option 2.")
  print("Option 0: Quit")
  print("Option 1: Encode a string")
  print("Option 2: Decode a string\n")

def translate_message(message_list, output_list):
  message = ""
  for i in range(len(message_list)):
    for j in range(len(output_list)):
      if ( abs(message_list[i] - output_list[j]) <= 0.001):
        message += chr(j + 32)
  print(message + '\n')

def message_to_matrix(message):
  matrix = []
  for i in range(len(message)):
    temp = bin(int(binascii.hexlify(message[i]), 16) - 32)
    temp = temp[2:]
    for j in range(width - len(temp)):
      temp = '0' + temp
    matrix.append(string_to_vect(temp))
  return matrix

def forward(W, B, message):
  user_input_matrix = message_to_matrix(message)
  forward_array = []
  # Build the graph
  tf.reset_default_graph()
  X = tf.placeholder(dtype=tf.float32)
  Z = tf.reduce_sum(tf.tanh(W * X + B))
  init = tf.global_variables_initializer()
  # Define and run the session
  with tf.Session() as sess:
    sess.run(init)
    for i in range(len(user_input_matrix)):
      out = sess.run(Z, feed_dict={X: user_input_matrix[i]})
      forward_array.append(out)
  return forward_array

inputs_encode, outputs_encode = generate_input_output()
dec_to_vect, bin_to_dec = generate_dictionaries(inputs_encode)

inputs_encode.reverse()

inputs_decode, outputs_decode = flip_mapping(inputs_encode, outputs_encode, dec_to_vect, bin_to_dec)

print("Mapping for encoding")
print_mapping(inputs_encode)
print("\n\nMapping for decoding")
print_mapping(inputs_decode)


# Build the graph
tf.reset_default_graph()
X_encode = tf.placeholder(dtype=tf.float32)
W_encode = tf.Variable(tf.random_normal(shape=[width, width],mean=0.5, stddev=1.0, dtype=tf.float32), dtype=tf.float32)
B_encode = tf.Variable(tf.zeros(shape=[width, 1], dtype=tf.float32))
Z_encode = tf.reduce_sum(tf.tanh(W_encode * X_encode + B_encode))
#Z_encode = tf.reduce_mean(tf.tanh(W_encode * X_encode + B_encode))

X_decode = tf.placeholder(dtype=tf.float32)
W_decode = tf.Variable(tf.random_normal(shape=[width, width],mean=0.5, stddev=1.0, dtype=tf.float32), dtype=tf.float32)
B_decode = tf.Variable(tf.zeros(shape=[width, 1], dtype=tf.float32))
Z_decode = tf.reduce_sum(tf.tanh(W_decode * X_decode + B_decode))

init = tf.global_variables_initializer()

# Loss & Optimizer
Y_encode = tf.placeholder(dtype=tf.float32)
loss_encode = tf.losses.mean_squared_error(Y_encode, Z_encode)
optimizer_encode = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_encode)

Y_decode = tf.placeholder(dtype=tf.float32)
loss_decode = tf.losses.mean_squared_error(Y_decode, Z_decode)
optimizer_decode = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_decode)

# Define and run the session
with tf.Session() as sess:
  sess.run(init)
  for epoch in range(num_epochs):
    total_loss_encode = 0
    total_loss_decode = 0
    if (epoch % 100 == 0):
      print("Epoch {}/{}".format(epoch + 1, num_epochs))
    for i in range(N):
      _, out_encode, out_loss_encode = sess.run([optimizer_encode, Z_encode, loss_encode], feed_dict={X_encode: inputs_encode[i], Y_encode: outputs_encode[i]})
      total_loss_encode += out_loss_encode
      _, out_decode, out_loss_decode = sess.run([optimizer_decode, Z_decode, loss_decode], feed_dict={X_decode: inputs_decode[i], Y_decode: outputs_decode[i]})
      total_loss_decode += out_loss_decode
      if (epoch == num_epochs - 1):
        print("Encode: Entry {}: (Expected: {} Predicted: {})".format(i, outputs_encode[i], out_encode))
        print("Decode: Entry {}: (Expected: {} Predicted: {})".format(i, outputs_decode[i], out_decode))
    if (epoch % 100 == 0):
      print("Encode: Total loss: {:0.6f}".format(total_loss_encode))  
      print("Decode: Total loss: {:0.6f}".format(total_loss_decode))
  weights_encode = sess.run(W_encode)
  biases_encode = sess.run(B_encode)
  weights_decode = sess.run(W_decode)
  biases_decode = sess.run(B_decode)



encode_list = []
decode_list = []
print('\n')
print_options()
option = raw_input()
while (option != '0'):
  if (option == '1'):
    user_input = raw_input("Enter message to encode.\n")
    encode_list = forward(weights_encode, biases_encode, user_input)
    translate_message(encode_list, outputs_encode)
  elif (option == '2'):
    user_input = raw_input("Enter message to decode.\n")
    decode_list = forward(weights_decode, biases_decode, user_input)
    translate_message(decode_list, outputs_decode)
  print_options()
  option = raw_input()


