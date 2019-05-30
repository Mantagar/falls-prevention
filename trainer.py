from model_utils import *
import sys
import time

hiddenSize=int(sys.argv[1])
depth=int(sys.argv[2])
bidirectional=(int(sys.argv[3])==1)
sequenceLength=300
minibatchSize=16
lr=1
lrd=0.97

dataPaths, inputSize, outputSize = loadDataPaths("data/training_set.txt")

testDataPaths, _, _ = loadDataPaths("data/test_set.txt")
testDataPaths2, _, _ = loadDataPaths("data/validation_set.txt")
testDataPaths += testDataPaths2

modelName = str(hiddenSize)+"_"+str(depth)+"_"+str(bidirectional)+"_"+str(int(time.time()))
model = RNN(inputSize, hiddenSize, depth, outputSize, bidirectional).double()
  
batcher = Batcher(dataPaths, sequenceLength, minibatchSize)

epochNum = 0
while True:
  lr = trainModel(model, batcher, maxEpochs=1, learningRate=lr, learningRateDecay=lrd, printLoss=True)
  '''
  df = testModel(model, dataPaths)
  df.to_csv("csv/"+modelName+'_'+str(epochNum)+'_training.csv', index=False)
  '''
  df = testModel(model, testDataPaths)
  df.to_csv("csv/"+modelName+'_'+str(epochNum)+'_test.csv', index=False)
  torch.save(model, 'checkpoints/'+modelName+'_'+str(epochNum)+".model")
  epochNum += 1