import forward_prop
import backward_prop
import tensorflow as tf


def main():

    input = "abcdefghijklmnop" #input string - can be anything for encoding
    N = 26  # Number of data
    width = 5  # Data width
    inputs = []
    for i in range(N):
        temp = []
        curVal = i
        for j in range(width):
            temp.append(curVal % 2)
            curVal = curVal / 2
        temp.reverse()
        inputs.append(temp)

    outputs = []
    for i in range(1, N + 1):
        outputs.append(i / float(N))
    outputs.reverse()
    back_weights = Backward_Propogation(inputs, outputs, N, width)
    print(back_weights)

    #call backward_prop - input is alphabet of strings; return weights (encode) 
    #call forward_prop function - returns output (encode)
    #print encoded message

    #call backward_Prop - input is output of forward_prop of encoding; returns weights (decode)
    #call forward_prop - returns output (decode) 

    #print decoded message statements

main()
