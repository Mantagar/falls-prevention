from skopt import gp_minimize
from model_utils import *
from skopt import load, dump
import sys
import time

def errorRate(x):
  [hiddenSize, depth, sequenceLength] = x
  hiddenSize = int(hiddenSize)
  depth = int(depth)
  sequenceLength = int(sequenceLength)
  minibatchSize = 16
  lr = 1.0
  lrd = 0.9
  threshold = 0.7
  
  dataPaths, inputSize, outputSize = loadDataPaths("Processed data/training_set.txt")
  model = RNN(inputSize, hiddenSize, depth, outputSize).double()
  batcher = Batcher(dataPaths, sequenceLength, minibatchSize)
  trainModel(model, batcher, maxEpochs=5, learningRate=lr, learningRateDecay=lrd, printLoss=False)
  validationDataPaths, _, _ = loadDataPaths("Processed data/validation_set.txt")
  df = testModel(model, validationDataPaths)
  
  return 1. - getAccuracy(df, threshold)

class CheckpointSaver(object):
  def __init__(self, name):
    self.name = name
    
  def __call__(self, res):
     dump(res, 'checkpoints/'+self.name+'.tuner', store_objective=False)

name = str(int(time.time()))
argc = len(sys.argv)
if argc>1:
  x0 = []
  y0 = []
  for i in range(1,argc):
    res = load('checkpoints/'+sys.argv[i]+'.tuner')
    x0.extend(res.x_iters)
    y0.extend(res.func_vals)
  args = {'x0': x0, 'y0': y0, 'n_random_starts': 0}
else:
  args = {'n_random_starts': 10}
  
res = gp_minimize(errorRate, [(50,200), (1,2), (50,300)], n_calls=50, callback=[CheckpointSaver(name)], **args)

print(res.x)
print(res.fun)
print("Iteration no:", len(res.func_vals))

