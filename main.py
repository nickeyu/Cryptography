import forward_prop
import backward_prop
import tensorflow as tf
import operator

def main():

    alphabet = "abcdefghijklmnopqrstuvwxyz" #input string - can be anything for encoding
    N = len(alphabet)  # Number of data
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

    mapper = forward_prop.Forward_Propagation(inputs, N, width, 0, 0, 1)

#    print(mapper)

#    print(outputs)

#    [k for k, v in sorted(zip(mapper, inputs), key=operator.itemgetter(1))]

    for i in range(N):
        min_temp_index = i
        for j in range(i, N):
            if(mapper[j] < mapper[min_temp_index]):
                min_temp_index = j
        mapper[i], mapper[min_temp_index] = mapper[min_temp_index], mapper[i]
        outputs[i], outputs[min_temp_index] = outputs[min_temp_index], outputs[i]

#    print(mapper)

#    print(outputs)

    flag = 1
    while(True):
        if flag:
            flag = 0
            inputs = []
        if inputs != []:
            break
        inputs2 = raw_input("enter string: ")
        for i in inputs2:
            temp = []
            if(alphabet.find(i) == -1):
                flag = 1
            else:
                inputs3 = alphabet.find(i)
                for j in range(width):
                    temp.append(inputs3 % 2)
                    inputs3 = inputs3 / 2
                temp.reverse()
                inputs.append(temp)
    N = len(inputs)
    print(inputs)
    back_weights, labels = backward_prop.Backward_Propagation(inputs, outputs, N, width)
#    print(back_weights)

    forward_prop.Forward_Propagation(inputs, N, width, back_weights, labels, 0)

    #call backward_prop - input is alphabet of strings; return weights (encode) 
    #call forward_prop function - returns output (encode)
    #print encoded message

    #call backward_Prop - input is output of forward_prop of encoding; returns weights (decode)
    #call forward_prop - returns output (decode) 

    #print decoded message statements

main()
