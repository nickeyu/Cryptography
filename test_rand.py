#import forward_prop
import forward_prop_hidden
#import backward_prop
import backward_prop_hidden
import tensorflow as tf
import operator
import numpy as np
import random
import binascii

from tensorflow.python.client import device_lib
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

N = 95  # Number of data
width = 7#N  # Data width

def print_mapping(input_array):
  for i in range(len(input_array)):
    print("{} --> {} ".format(input_array[i], i))


def string_to_vect(input_string):
  temp_array = []
  for i in range(len(input_string)):
    temp_array.append(int(input_string[i]))
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
#  for i in range(width):
#    temp.append(0)
  for i in range(N):
    outputs.append(float(i + 1) / float(N))
    binary = '{0:b}'.format(i)
    for j in range(width):
      if j < width - (len(binary)):
        temp.append(0)
      else:
        temp.append(int(binary[-(width - j)]))
    inputs.append(temp)
#    inputs[i][i] = 1
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
  print("\nTranslated message: " + message + '\n')

def message_to_matrix(message):
  matrix = []
  for i in range(len(message)):
    #temp = bin(int(binascii.hexlify(message[i]), 16) - 32) #before python3
    temp = bin(int.from_bytes(message[i].encode(), 'big') - 32) #python 3.2+
    temp = temp[2:]
    for j in range(width - len(temp)):
      temp = '0' + temp
    matrix.append(string_to_vect(temp))
  return matrix


def main():
  inputs_encode, outputs_encode = generate_input_output()
  dec_to_vect, bin_to_dec = generate_dictionaries(inputs_encode)
  random.shuffle(inputs_encode)
  inputs_decode, outputs_decode = flip_mapping(inputs_encode, outputs_encode, dec_to_vect, bin_to_dec)  
  
  print(inputs_decode)
  print(outputs_decode)
  encode_weights_hidden, encode_biases_hidden, encode_weights, encode_biases, encode_loss = backward_prop_hidden.Backward_Propagation(inputs_encode, outputs_encode, N, width, 0, 0, 0, 0, 0)
  while(encode_loss > 0.001):
    encode_weights_hidden, encode_biases_hidden, encode_weights, encode_biases, encode_loss = backward_prop_hidden.Backward_Propagation(inputs_encode, outputs_encode, N, width, encode_weights_hidden, encode_biases_hidden, encode_weights, encode_biases, 1)

  decode_weights_hidden, decode_biases_hidden, decode_weights, decode_biases, decode_loss = backward_prop_hidden.Backward_Propagation(inputs_decode, outputs_decode, N, width, 0, 0, 0, 0, 0)
  while(decode_loss > 0.001):
    decode_weights_hidden, decode_biases_hidden, decode_weights, decode_biases, decode_loss = backward_prop_hidden.Backward_Propagation(inputs_decode, outputs_decode, N, width, decode_weights_hidden, decode_biases_hidden, decode_weights, decode_biases, 1)

  print('\n')
  print_options()
  #option = raw_input()
  option = input()
  while (option != '0'):
    if (option == '1'):
      #user_input = raw_input("Enter message to encode.\n")
      user_input = input("Enter a message to encode.\n")
      print()
      user_input_vectors = message_to_matrix(user_input)
      encode_predictions = forward_prop_hidden.Forward_Propagation(user_input_vectors, len(user_input_vectors), width, encode_weights_hidden, encode_biases_hidden, encode_weights, encode_biases, 0)
      #enc_temp = []
      #for i in encode_predictions:
      #   for j in i: #look through list in list
      #       for k in j: #look through arrays in list   
      #           temp = (k)
      #           print ("Shape: ", tf.shape(temp))
      #           print (temp)
      #           enc_temp.append(temp)  
      #print(enc_temp)
      #for i in enc_temp:
      #    print(i)
      #print(outputs_encode)
      translate_message(encode_predictions, outputs_encode)
    elif (option == '2'):
      #user_input = raw_input("Enter message to decode.\n")
      user_input = input("Enter a message to decode.\n")
      print()
      user_input_vectors = message_to_matrix(user_input)
      decode_predictions = forward_prop_hidden.Forward_Propagation(user_input_vectors, len(user_input_vectors), width, decode_weights_hidden, decode_biases_hidden, decode_weights, decode_biases, 0)
      #dec_temp = []
      #for i in decode_predictions:
         #print(i)
      #   for j in i:
      #       for k in j:
      #          dec_temp.append(k)
      #          print (k)
      translate_message(decode_predictions, outputs_decode)
    print_options()
    #option = raw_input()
    option = input()


main()
