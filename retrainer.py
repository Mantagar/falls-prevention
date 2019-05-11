from model_utils import *
import sys
import time

sequenceLength=300
minibatchSize=16
lr=1
lrd=0.9

dataPaths, inputSize, outputSize = loadDataPaths("data/training_set.txt")

modelName = sys.argv[1]
model = torch.load('checkpoints/'+modelName+'.model')
  
batcher = Batcher(dataPaths, sequenceLength, minibatchSize)

epochNum = 0
while True:
  lr = trainModel(model, batcher, maxEpochs=1, learningRate=lr, learningRateDecay=lrd, printLoss=True)
  
  df = testModel(model, dataPaths)
  df.to_csv("csv/"+modelName+'_'+str(epochNum)+'_training.csv', index=False)
  torch.save(model, 'checkpoints/'+modelName+'_'+str(epochNum)+".model")
  epochNum += 1