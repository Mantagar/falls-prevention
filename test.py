from model_utils import *
import sys

trainingDataPaths = loadListFromFile("Processed data/training_set.txt")
validationDataPaths = loadListFromFile("Processed data/validation_set.txt")

model = torch.load(sys.argv[1])
print("TRAINING LOSS\t\t\tVALIDATION LOSS")
print(str(calculateAverageLoss(model, trainingDataPaths)) + "\t\t" + str(calculateAverageLoss(model, validationDataPaths)))