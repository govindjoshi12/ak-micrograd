import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Helper Methods

def topologicalSort(root):

    topo = []
    visited = set()

    # recursive dfs
    def topoHelper(node, topo, visited):
        if node in visited:
            return 
        visited.add(node)

        for c in node._prev:
            topoHelper(c, topo, visited)
        topo.append(node)

    topoHelper(root, topo, visited)

    # returns nodes in reverse topologically sorted order
    return topo

# Node in computational graph
# children combine to produce current Value
class Value:

    # input nodes have no children and no operator,
    # They are the 'leaves' of the computational graph
    
    # I believe we are using this "backwards" structure
    # where inputs are leaves and final outputs are roots
    # because we are preparing for backprop
    def __init__(self, data, _children=(), _op="", label=''):
        self.data = data
        self.grad = 0.0

        # default _backward function, ex. for a leaf, does nothing
        # We set the backward functions of parent nodes when creating them.
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op 
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    # Operators

    # Allow operating on constants (floats that are not wrapped in Value objects)
    # If val is not of type Value, assumes that it is a float
    def processOperand(self, val):
        return val if isinstance(val, Value) else Value(val)

    # a + b
    # left value will be self, right value will be other
    def __add__(self, val):
        val = self.processOperand(val)
        out = Value(self.data + val.data, (self, val), '+')

        # When parent calls _backward, its gradient is propagated
        # to us (its' children), and we update our gradients
        # using the chain rule. 
        def _backward():
            # local derivative * dFinal / dOut
            # Addition function: simply "route" parent gradient to me.
            # We do += because this node could be involved in other calculations
            # and thus could have multiple parents. The partial derivative in that case
            # is the sum of each dParent/dChild derivative. 
            self.grad += 1.0 * out.grad  
            val.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    # right add to constant. Ex. 2 + a
    def __radd__(self, val):
        return self + val
        
    def __sub__(self, val):
        val = self.processOperand(val)
        out = Value(self.data - val.data, (self, val), '-')

        # Similar to __add__, but slight differences
        # next = self - val
        # deriv w.r.t val = -1
        # deriv w.r.t self = 1
        def _backward():
            self.grad += 1.0 * out.grad  
            val.grad += -1.0 * out.grad
        out._backward = _backward

        return out

    # right subtract from constant. Ex. 2 - a
    def __rsub__(self, val):
        return Value(val) - self

    def __mul__(self, val):
        val = self.processOperand(val)
        out = Value(self.data * val.data, (self, val), '*')

        def _backward():
            self.grad += val.data * out.grad
            val.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    # Ex. 2 * a
    def __rmul__(self, val):
        return self * val
    
    # Only supporting int/floats for val for now
    # TODO: Support Value objects
    def __pow__(self, val):
        assert(isinstance(val, (int, float)))

        out = Value(self.data ** val, (self,), f'**{val}')

        # power rule
        def _backward():
            self.grad += (val * (self.data ** (val - 1))) * out.grad
        out._backward =_backward
        
        return out

    # python3 uses __truediv__ for / 
    #          and __floordiv__ for //
    # Division implemented with __pow__ operator
    def __truediv__(self, val):
        out = self * (val ** -1.0)
        out._op = '/'
        return out

    # Originally implemented division operator.
    def atomic_div(self, val):
        val = self.processOperand(val)
        out = Value(self.data / val.data, (self, val), '/')

        # Similar to __mul__ but slight differences
        # next = self * (val)^-1
        # dNext/dVal = -1 * self * (val)^-2
        # dNext/dSelf = 1 / val
        def _backward():
            self.grad += (1.0 / val.data) * out.grad
            val.grad += (-1.0 * self.data * (val.data ** -2.0)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        val = math.exp(self.data)
        out = Value(val, (self,), 'exp')
        
        # derivative of e^x = e^x
        # next = exp(self)
        # dNext/dSelf = exp(self)
        def _backward():
            self.grad += val * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        expVal = (self * 2.0).exp()
        val = (expVal - 1.0) / (expVal + 1.0)

        # Each atomic function used in the expression above will 
        # provided its own _backward function, with division 
        # being the first function differntiated during backprop
        return val  

    def relu(self):
        outVal = self.data if self.data > 0 else 0
        out = Value(outVal, (self,), 'relu')

        def _backward():
            self.grad = (1 if outVal else 0) * out.grad
        out._backward = _backward

        return out

    # We can write out the tanh function explicitly by implementing
    # __sub__, __div__, and exp. But it's not necessary to have 
    # these atomic computational pieces. You can have something as 
    # simple as an addition operator or something arbitrarily complex, 
    # but the most important thing is that you know how to differentiate
    # the function you implement so you compute the "local" partial derivative.
    def atomic_tanh(self):
        x = self.data
        val = (math.exp(2.0 * x) - 1.0) / (math.exp(2.0 * x) + 1.0)
        out = Value(val, (self,), 'tanh')
        
        def _backward():
            self.grad += (1.0 - val ** 2.0) * out.grad
        out._backward = _backward
        return out
    
    # I can implement this with atomic operations I've already written
    def squaredDist(self, val):
        val = self.processOperand(val)
        out = (val - self) ** 2
        return out

    def sign(self):
        outVal = 1 if self.data > 0 else 0
        out = Value(outVal, (self,), 'sign')

        # don't use this function for gradient descent 
        # because it has a 0 derivative. 
        def _backward():
            self.grad = 0
        out._backward = _backward

        return out

    
    # --- static functions for non-linearities ---
    # my convention: capitalize first letter for 
    # static activation functions 

    @staticmethod
    def Tanh(value: 'Value'):
        return value.tanh()

    @staticmethod
    def Relu(value: 'Value'):
        return value.relu()
    
    @staticmethod
    def Sign(value: 'Value'):
        return value.sign()

    @staticmethod
    # useful for implementing linear regression
    def Identity(value: 'Value'):
        return value


    # --- AUTOGRAD ---
    
    def backward(self):
        # can run autograd with any node being the "global" function
        # against which each variable is differentiated. Just need to 
        # set my grad to 1 (base case) before running backward
        
        self.grad = 1 # base case
        rev = topologicalSort(self)
        for i in range(len(rev) - 1, -1, -1):
            rev[i]._backward()

# TODO:
# Would be very cool to implement more operators, such as log, sigmoid,
# ReLU, etc. and other non-linearities, along with things like self-attention