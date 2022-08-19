import random
import time
import numpy as np
import math
import copy

total_time = time.time()  # starts a timer


class i_node:  # input node
    def __init__(self, value, nextlayersize):
        self.weights = [0.0 for i in range(
            nextlayersize)]  # sets a random weight(-1.00, 1.00) for all the synapses spreading out from the input node
        self.value = value


class h_node:  # hidden layer node
    def __init__(self, nextlayersize):
        self.weights = [0.0 for i in range(nextlayersize)]
        self.bias = 0
        self.value = 0


class o_node:  # output node
    def __init__(self, number):
        self.number = number
        self.value = 0  # confidence that the ai has in its decision
        self.bias = 0


NN = []  # this list will store the

weights = []
values = []
bias = []


def init(inputs):
    inputlayer = []
    outputlayer = []
    for i in range(NN_layout[0]):
        inputlayer.append(i_node(inputs[i], NN_layout[1]))
    NN.append(inputlayer)
    for i in range(len(NN_layout) - 2):
        NN.append([h_node(NN_layout[i + 2]) for x in range(NN_layout[i + 1])])
    for i in range(NN_layout[-1]):
        outputlayer.append(o_node(i))
    NN.append(outputlayer)


def adjust_modifiers(NN, variability):
    for i in range(NN_length - 2):
        for node in NN[i + 1]:
            for i, weight in enumerate(node.weights):
                node.weights[i] += round(random.uniform(variability * -1, variability), 2)
            node.bias += round(random.uniform(variability * -1, variability), 2)

    for node in NN[0]:
        for i, weight in enumerate(node.weights):
            node.weights[i] += round(random.uniform(variability * -1, variability), 2)
    for node in NN[-1]:
        node.bias += round(random.uniform(variability * -1, variability), 2)


def display_NN():  # this function is not required for the program to run and is only
    print("inputlayer")
    for node in NN[0]:
        print(node.value, node.weights)
    print("-----------------------------")
    for i in range(len(NN_layout) - 2):
        print("hiddenlayer:", i)
        for node in NN[i + 1]:
            print(i, node.value, node.weights, node.bias)
    print("-----------------------outputlayer-----------------------")
    for node in NN[-1]:
        print(node.number, node.value)
    print("")


def sigmoid(x):  # squashes a number between 0, and 1

    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig


def dotproduct(values, weights, bias):
    return np.dot(weights, values) + bias


def cost(expected, actual):
    total = 0
    for i, a in enumerate(expected):
        if a == 1:
            total += ((actual[i] - 1) ** 2)

        elif a == 0:
            total += ((actual[i] - 0) ** 2)

    if expected.index(max(expected)) == actual.index(max(actual)):
        return total
    else:
        return total


def foward_propagate(set):
    for n in range(NN_length - 1):
        values = np.array([set[n][i].value for i in range(len(set[n]))])
        weights = np.array([set[n][i].weights for i in range(len(set[n]))]).transpose()
        bias = np.array([set[n + 1][i].bias for i in range(len(set[n + 1]))])

        result = dotproduct(values, weights, bias)
        for i, v in enumerate(result):
            set[n + 1][i].value = sigmoid(v)
    output = [set[-1][x].value for x in range(len(set[-1]))]


def get_output(set):
    values = []
    for node in set[-1]:
        values.append(node.value)

    return values

def append_to_file(set):
    file_object = open('Ai.txt', 'a')
    file_object.write('---Inputlayer---')
    file_object.write('\n')

    file_object.write("values")
    file_object.write('\n')
    file_object.write(str([node.value for node in set[0]]))
    file_object.write('\n')

    file_object.write('Weights')
    file_object.write('\n')
    for i, node in enumerate(set[0]):
        file_object.write(str(i))
        file_object.write(' : ')
        file_object.write(str(node.weights))
        file_object.write('\n')



    for n in range(NN_length-2):
        file_object.write('\n')
        file_object.write('\n')
        file_object.write('--Hiddenlayer--: ')
        file_object.write(str(n+1))
        file_object.write('\n')

        file_object.write('Values')
        for i, node in enumerate(set[n + 1]):
            file_object.write(str(i))
            file_object.write(' : ')
            file_object.write(str(node.value))
            file_object.write('\n')

        file_object.write('Weights')
        for i, node in enumerate(set[n+1]):
            file_object.write(str(i))
            file_object.write(' : ')
            file_object.write(str(node.weights))
            file_object.write('\n')

        file_object.write('\n')
        file_object.write('Bias')
        file_object.write('\n')

        for i, node in enumerate(set[n+1]):
            file_object.write(str(i))
            file_object.write(' : ')
            file_object.write(str(node.bias))
            file_object.write('\n')

    file_object.write('\n')
    file_object.write('\n')
    file_object.write('\n')
    file_object.write('---OutputLayer---')

    file_object.write('\n')
    file_object.write('\n')

    file_object.write('Values')
    file_object.write('\n')
    for i, node in enumerate(set[-1]):
        file_object.write(str(i))
        file_object.write(' : ')
        file_object.write(str(node.value))
        file_object.write('\n')

    file_object.write('Bias')
    for i, node in enumerate(set[-1]):
        file_object.write(str(i))
        file_object.write(' : ')
        file_object.write(str(node.bias))
        file_object.write('\n')

    file_object.close()

def show_result(set):
    file_object = open('Ai.txt', 'a')
    file_object.write('---Inputlayer---')
    file_object.write('\n')

    file_object.write("values")
    file_object.write('\n')
    file_object.write(str([node.value for node in set[0]]))
    file_object.write('\n')
    file_object.write('\n')

    file_object.write('Output')
    file_object.write('\n')
    for i, node in enumerate(set[-1]):
        file_object.write(str(i))
        file_object.write(' : ')
        file_object.write(str(node.value))
        file_object.write('\n')
    file_object.write('\n')
    file_object.write('\n')
    file_object.write('\n')
    file_object.write('\n')


NN_layout = [3, 10,
             1]  # first number is the size of the input layer, last number is the size of the output layer, all other values are the size of hidden layers
NN_length = len(NN_layout)

truth_table = {
    '0,1,0': "1",
    '3,1,0': "1",
    '1,1,2': "1",
    '0,3,0': "1",
    '0,1,1': "1",
    '3,2,2': "1",
    '1,2,2': "1",

    '0,2,0': "0",
    '2,2,0': "0",
    '0,0,2': "0",
    '2,2,2': "0",
    '2,0,0': "0",
    '0,2,2': "0",
    '0,0,0': "0"
}


'''
truth_table = {
    '1,0': "1",
    '5,2': "1",
    '4,2': "1",
    '1,1': "1",
    '4,0': "1",
    '3,2': "1",
    '4,1': "1",
    '2,1': "1",
    '3,0': "1",
    '5,4': "1",
    '1,2': "1",
    '1,4': "1",

    '-1,0': "0",
    '5,-2': "0",
    '-4,2': "0",
    '-1,1': "0",
    '-4,0': "0",
    '3,-2': "0",
    '-4,1': "0",
    '2,-1': "0",
    '-3,0': "0",
    '-5,4': "0",
    '1,-2': "0",
    '-1,4': "0",
}
'''


sample_size = 50
generations = 1000
iterations = 100
learning_rate = 1000
generationCost = []
fcost = 1
best_cost = 5

for x in range(iterations):

    print("iteration complete")
    for y in range(generations):
        generationCost = []
        if y == 0 and x == 0:
            given_inputs = random.choice(list(truth_table.keys()))
            given_inputs = given_inputs.split(",")
            given_inputs = [int(x) for x in given_inputs]

            init(given_inputs)
            adjust_modifiers(NN, 1)


        currentGenNN = copy.deepcopy(NN)
        adjust_modifiers(currentGenNN, best_cost*learning_rate)

        for k in range(sample_size):

            given_inputs = random.choice(list(truth_table.keys()))
            given_inputs = given_inputs.split(",")
            given_inputs = [int(x) for x in given_inputs]

            if k != 0:
                for a, l in enumerate(currentGenNN[0]):
                    l.value = given_inputs[a]

            foward_propagate(currentGenNN)
            output = get_output(currentGenNN)

            inputstr = ",".join(str(e) for e in given_inputs)
            correct = truth_table.get(str(inputstr))

            if correct == "1":
                fcost = round(int(correct) - output[0], 4)
            elif correct == "0":
                fcost = round((output[0] - int(correct)), 4)

            generationCost.append(fcost)
        averageCost = sum(generationCost)/len(generationCost)

        if averageCost < best_cost:
            BestNN = copy.deepcopy(currentGenNN)
            best_cost = averageCost


    NN = copy.deepcopy(currentGenNN)
    print(best_cost)
    best_cost = 10



print("code took: ", time.time() - total_time, "seconds to run") 