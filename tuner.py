from skopt import gp_minimize
from model_utils import *
from skopt import load, dump
import sys

def errorRate(x):
  [hiddenSize, depth, sequenceLength, minibatchSize, lr, lrd, threshold] = x
  hiddenSize = int(hiddenSize)
  depth = int(depth)
  sequenceLength = int(sequenceLength)
  minibatchSize = int(2 ** minibatchSize)
  lr = 10 ** lr
  
  dataPaths, inputSize, outputSize = loadDataPaths("Processed data/validation_set.txt")
  model = RNN(inputSize, hiddenSize, depth, outputSize).double()
  batcher = Batcher(dataPaths, sequenceLength, minibatchSize)
  trainModel(model, batcher, maxEpochs=3, learningRate=lr, learningRateDecay=lrd, printLoss=False)
  df = testModel(model, dataPaths)
  
  return 1. - getAccuracy(df, threshold)
  
def saveCheckpoint(res):
  dump(res, 'checkpoint.tuner', store_objective=False)


if len(sys.argv)>1:
  res = load(sys.argv[1])
  args = {'x0': res.x_iters, 'y0': res.func_vals, 'n_random_starts': 0}
else:
  args = {'n_random_starts': 10}
  
res = gp_minimize(errorRate, [(50,200), (1,3), (50,300), (0,6), (-3.,0.), (0.8, 0.99), (0.5,1.)], n_calls=20, callback=saveCheckpoint, **args)

print(res.x)
print(res.fun)
print("Iteration no:", len(res.func_vals))

