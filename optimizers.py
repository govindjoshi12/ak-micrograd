class Optimizer:

    # params is a list of Value objects
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    
    def step(self):
        for param in self.params:
            # move in opposite direction of gradient
            param.data -= self.lr * param.grad
    
    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0