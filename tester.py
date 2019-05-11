from model_utils import *
import sys

  
dataPaths, _, _ = loadDataPaths("data/test_set.txt")
dataPaths2, _, _ = loadDataPaths("data/validation_set.txt")
dataPaths += dataPaths2

name = sys.argv[1]

model = torch.load('checkpoints/'+name+'.model')

df = testModel(model, dataPaths)

df.to_csv("csv/"+name+'_test.csv', index=False)
