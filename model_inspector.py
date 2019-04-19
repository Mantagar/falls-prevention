from model_utils import *
import sys

name = sys.argv[1]

model = torch.load('checkpoints/'+name+'.model')
print("Input size:\t" + str(model.input_size))
print("Hidden size:\t" + str(model.hidden_size))
print("Depth:\t\t" + str(model.stacks))
print("Output size:\t" + str(model.output_size))