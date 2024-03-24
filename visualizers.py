from graphviz import Digraph
from collections import deque
from nn import MLP

### VISUALIZATION

# --- Visualize computation graph ---
def getGraph(root):

    adjList = {}

    frontier = deque([root])
    while frontier:
        node = frontier.pop()
        if not node:
            continue
        adjList[node] = set()

        for c in node._prev:
            adjList[node].add(c)
            frontier.append(c)
    
    return adjList

def graphVisualizer(root):

    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) 
    # LR: left-to-right (default is vertical)
    
    adjList = getGraph(root)
    for node in adjList:
        # shape: record - uses rectangle for data. Without specifiying, defaults to ovals for operators
        # { %s | ... } I think this is special syntax for graphviz to obtain the bar separator 
        dot.node(str(id(node)), 
                "{ %s | data: %.4f | grad: %.4f  }" % (node.label, node.data, node.grad), 
                shape="record")

    # The actual adjList is "backwards": the final output is the parent.
    # We want to display the forward pass, so we will flip the edges
    for node, children in adjList.items():
        # id(obj) returns unique integer identifier for obj
        uid = str(id(node))
        if node._op:
            opNodeId = uid + node._op
            dot.node(opNodeId, node._op)
            dot.edge(opNodeId, uid)
            for c in children:
                dot.edge(str(id(c)), opNodeId)

    return dot

### --- Visualize MLP ---

def visualizeMlp(mlp: MLP):

    dot = Digraph(format='svg', graph_attr={ 'rankdir': 'LR' })

    # 1. Add all neurons to the graph
    for layerIdx, layer in enumerate(mlp.layers):
        for neuronIdx, neuron in enumerate(layer.neurons):
            nodePrefix = 'out' if layerIdx == len(mlp.layers) - 1 else 'n'
            label = f'{nodePrefix}_{layerIdx},{neuronIdx}'
            dot.node(str(id(neuron)), label=label)
    
    for i in range(len(mlp.layers) - 1, 0, -1):
        currLayer = mlp.layers[i]
        prevLayer = mlp.layers[i-1]

        for neuronIdx, neuron in enumerate(currLayer.neurons):
            for prevNeuronIdx, prevNeuron in enumerate(prevLayer.neurons):
                dot.edge(str(id(prevNeuron)), str(id(neuron)), 
                         # f-string truncate float: f'num={value:.2f}'
                         f'w_{neuronIdx},{prevNeuronIdx}')
    
    # There aren't actually any neurons in the network representing the input
    # We'll visualize the input here
    for inputIdx in range(mlp.layers[0].inputDim):
        nodeId = f'{inputIdx}' + str(id(mlp.layers[0]))
        dot.node(nodeId, label=f'inp_{inputIdx}')
        for neuronIdx, neuron in enumerate(mlp.layers[0].neurons):
            dot.edge(nodeId, str(id(neuron)), label=f'w_{neuronIdx},{inputIdx}')

    
    return dot
