from model_utils import *
import sys

  
testDataPaths = loadListFromFile("Processed data/training_set.txt")

name = sys.argv[1]

model = torch.load('checkpoints/'+name+'.model')

accuracy, df = test(model, testDataPaths)
print(accuracy)

df.to_csv(name+'.csv', index=False)
