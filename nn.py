import random
from typing import List
from micrograd import Value

# Interface for all the following modules
class Module:

    def __init__(self) -> None:
        pass

    def parameters(self) -> List:
        pass

# Single NN Neuron
class Neuron(Module):

    def __init__(self, numInputs, activation=Value.relu):

        self.weights = [Value(random.uniform(-1, 1)) for _ in range(numInputs)]
        self.bias = Value(random.uniform(-1, 1))
        self.activation = activation
    
    # forward(x)
    # x should be a list of ints/floats or Value() objects of size numInputs.
    def __call__(self, x):
        # assert(len(x) == len(self.weights))

        # zip returns an iterator of tuples: (self.weights[i], x[i]) pairs
        # sum() takes in an optional starting value
        rawValue = sum((w * x_i for w, x_i in zip(self.weights, x)), start=self.bias) 
        return self.activation(rawValue)

    # Identical to __call__, but implemented in an explicit, non-pythonic way
    # to clearly see the use of the Value operators we defined
    def explicit_call(self, x):
        assert(len(x) == len(self.weights))

        # Explicit (non-pythonic)
        rawValue = self.bias
        for i in range(len(self.weights)):
            # This uses the add and multiply operators in Value
            # (Not using += because we would need to define __iadd__ for that)
            rawValue = rawValue + self.weights[i] * x[i]
        return self.activation(rawValue)

    # Every module in pytorch has a parameters method that returns all the 
    # parameters of that module
    def parameters(self):
        return self.weights + [self.bias]
    
    def __repr__(self) -> str:
        return f'Neuron(numInputs={len(self.weights)}, act=tanh)'
    
class Layer(Module):
    
    def __init__(self, inputDim, outputDim, activation=Value.relu):
        # the output of each neuron call n(x) is a Value() object
        self.inputDim = inputDim
        self.neurons = [Neuron(inputDim, activation) for _ in range(outputDim)]
    
    # x should be list of Value objects with len(x) = inputDim
    # It represents the activations of the previous layer.
    # The list should either be the list of inputs, or a list generated a layer call
    def __call__(self, x):
        # if layer only has a single neuron, 
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        # nested for-loop list comprehension
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self) -> str:
        return f'Layer(inp={self.inputDim}, out={len(self.neurons)})'
    
class MLP(Module):

    # format based on pytorch MLP (using list of hiddenDims to specify MLP shape)
    def __init__(self, inputDim, hiddenDims=[5], activation=Value.relu):
        self.layers = []
        prevDim = inputDim
        for dim in hiddenDims:
            self.layers.append(Layer(prevDim, dim, activation))
            prevDim = dim
        
    # x should be a list numbers or Values of size inputDim
    def __call__(self, x):
        # Is there a pythonic way to do this? 
        transformed = x
        for layer in self.layers:
            transformed = layer(transformed)
        return transformed

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    
    def __repr__(self) -> str:
        return str(self.layers)

# MLP wrapper with 1 output layer of 1 dimension
class LinearRegressor(MLP):

    def __init__(self, inputDim, activation=Value.Identity):
        # super(): returns a delegate object to parent class
        super().__init__(inputDim, [1], activation)
    
# TODO: activation input validation?