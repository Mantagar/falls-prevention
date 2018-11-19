import torch
import pandas as pd
import random
import numpy

def loadListFromFile(path):
  with open(path, 'r') as file:
    list = file.readlines()
  return [ item.replace('\n', '') for item in list ]
  
def createMiniBatch(input, seq_index, seq_size, batch_size, input_size):
  #prepare a mini-batch of sequences (technically a sequence of mini-batches)
  batch = torch.from_numpy(input[seq_index:seq_index+seq_size]).reshape(seq_size, 1, input_size)
  seq_index += 1
  for i in range(1, batch_size):
    seq = torch.from_numpy(input[seq_index:seq_index+seq_size]).reshape(seq_size, 1, input_size)
    batch = torch.cat((batch, seq), 1)
    seq_index += 1
  return batch, seq_index

def calculateAverageLoss(model, dataPaths):
  loss_fn = torch.nn.CrossEntropyLoss()
  avg_loss = 0
  for path in dataPaths:
    df = pd.read_csv(path)
    input = df.values
    target = torch.LongTensor([0]) if "Nosynkope" in path else torch.LongTensor([1])
    batch, _ = createMiniBatch(input, 0, len(input), 1, model.input_size)
    pred = model(batch)[-1].view(1, model.output_size)
    avg_loss += loss_fn(pred, target).detach().numpy()
  avg_loss /= len(dataPaths)
  return avg_loss
  
class RNN(torch.nn.Module):
  def __init__(self, input_size, hidden_size, stacks, output_size):
    super(RNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.stacks = stacks
    self.output_size = output_size
    self.rnn = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=stacks)
    self.lastLayer = torch.nn.Linear(hidden_size, output_size)
    self.activFunc = torch.nn.Softmax(dim=2)
  
  def forward(self, batch):
    out, hidden_state = self.rnn(batch)
    out = self.lastLayer(out)
    return self.activFunc(out)
