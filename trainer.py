from model_utils import *
import sys

hiddenSize=100
depth=2
sequenceLength=100
minibatchSize=16
lr=1
lrd=0.9

dataPaths, inputSize, outputSize = loadDataPaths("Processed data/training_set.txt")

if len(sys.argv)>1:
  modelName = sys.argv[1]
  model = torch.load('checkpoints/'+modelName+'.model')
else:
  modelName = "network_"+str(random.randint(100000,1000000))
  model = RNN(inputSize, hiddenSize, depth, outputSize).double()
  
batcher = Batcher(dataPaths, sequenceLength, minibatchSize)

epochNum = 0
while True:
  lr = trainModel(model, batcher, maxEpochs=1, learningRate=lr, learningRateDecay=lrd, printLoss=True)
  
  df = testModel(model, dataPaths)
  df.to_csv(modelName+'_'+str(epochNum)+'_training.csv', index=False)
  torch.save(model, 'checkpoints/'+modelName+'_'+str(epochNum)+".model")
  epochNum += 1