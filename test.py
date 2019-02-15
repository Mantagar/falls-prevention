from model_utils import *
import sys

  
testDataPaths = loadListFromFile("Processed data/test_set.txt")

name = sys.argv[1]

model = torch.load('checkpoints/'+name+'.model')

df = test(model, testDataPaths)

df.to_csv(name+'_test.csv', index=False)
