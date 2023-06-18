import random
from engine.py import Value

class Neuron:

  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w, self.x)), self.b)
    out = act.tanh()
    return out

  def parameters(self):
    return self.w + [self.b]

class Layer:

  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs)==1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:

  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

# xs = [
#   [2.0, 3.0, -1.0],
#   [3.0, -1.0, 0.5],
#   [0.5, 1.0, 1.0],
#   [1.0, 1.0, -1.0],
# ]
# ys = [1.0, -1.0, -1.0, 1.0] # desired targets
# n = MLP(3, [4, 4, 1])

# for k in range(20):
  
#   # forward pass
#   ypred = [n(x) for x in xs]
#   loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
#   # backward pass
#   for p in n.parameters():
#     p.grad = 0.0
#   loss.backward()
  
#   # update
#   for p in n.parameters():
#     p.data += -0.1 * p.grad
  
#   print(k, loss.data)