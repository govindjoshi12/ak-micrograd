from optimizers import Optimizer
from nn import MLP
from collections import deque

def vprint(verbose=True, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

# simple MLP predict
def predict(model, xs):
    return [model(x) for x in xs]

# simple MLP training function 
# Works with Linear regression module as well
def train_mlp_mse(model: MLP, xs, ys, epochs=25, lr=0.01, verbose=True):

    optimizer = Optimizer(model.parameters(), lr)
    loss_per_epoch = []
    
    vprint(verbose, "Training...")
    for i in range(epochs):
        yPred = predict(model, xs)

        # batch loss
        # loss is just another layer in the computational graph!

        # taking mean doesn't matter in terms of gradient descent, but necessary
        # for loss to actually represent mse. (so divide by len(ys))
        loss = sum(pred[0].squaredDist(gold) for pred, gold in zip(yPred, ys)) / len(ys)
        loss_per_epoch.append(loss.data)

        if i % 5 == 0:
            vprint(verbose, f"Loss at epoch {i}: {loss.data}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss_per_epoch