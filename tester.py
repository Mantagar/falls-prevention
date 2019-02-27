from model_utils import *
import sys

  
dataPaths, _, _ = loadDataPaths("Processed data/test_set.txt")

name = sys.argv[1]

model = torch.load('checkpoints/'+name+'.model')

df = testModel(model, dataPaths)

df.to_csv(name+'_test.csv', index=False)
