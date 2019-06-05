from model_utils import *
import sys

name = sys.argv[1]

model = torch.load('checkpoints/'+name+'.model')
print("Input size:\t" + str(model.input_size), flush=True)
print("Hidden size:\t" + str(model.hidden_size), flush=True)
print("Depth:\t\t" + str(model.stacks), flush=True)
print("Output size:\t" + str(model.output_size), flush=True)
try:
  #Old RNN definition did not contain bidirectional field
  print("Bidirectional:\t" + str(model.bidirectional), flush=True)
except:
  print("Bidirectional:\tFalse", flush=True)