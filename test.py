from model_utils import *
import sys

  
testDataPaths = loadListFromFile("Processed data/test_set.txt")

model = torch.load(sys.argv[1])

accuracy = test(model, testDataPaths)
print(accuracy, flush=True)
