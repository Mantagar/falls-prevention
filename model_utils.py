import torch
import pandas as pd
import random
import numpy


class Batcher:
  def __init__(self, dataPaths, seq_size, batch_size):
    self.seq_size = seq_size
    self.batch_size = batch_size
    self.dataPaths = dataPaths
    self.data = []
    for path in dataPaths:
      values = pd.read_csv(path).values
      key = 1 if "Nosynkope" in path else 0
      seq_index = 0
      while seq_index+seq_size<len(values):
        self.data.append([key, values[seq_index:seq_index+seq_size]])
        seq_index += 1
    self.sample_amount = len(self.data)
    self.nextEpoch()
        
  def nextEpoch(self):
    data = random.sample(self.data, len(self.data))
    self.id = 0
  
  def hasNextBatch(self):
    return self.id + self.batch_size < self.sample_amount
    
  def nextBatch(self):
    batch_x = torch.from_numpy(self.data[self.id][1]).reshape(self.seq_size, 1, -1)
    batch_y = torch.LongTensor([self.data[self.id][0]]).repeat(self.seq_size).reshape(self.seq_size, 1)
    self.id += 1
    for i in range(self.batch_size-1):
      x = torch.from_numpy(self.data[self.id][1]).reshape(self.seq_size, 1, -1)
      y = torch.LongTensor([self.data[self.id][0]]).repeat(self.seq_size).reshape(self.seq_size, 1)
      batch_x = torch.cat((batch_x, x), 1)
      batch_y = torch.cat((batch_y, y), 1)
      self.id += 1
    return batch_x, batch_y

def loadListFromFile(path):
  with open(path, 'r') as file:
    list = file.readlines()
  return [ item.replace('\n', '') for item in list ]

class RNN(torch.nn.Module):
  def __init__(self, input_size, hidden_size, stacks, output_size):
    super(RNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.stacks = stacks
    self.output_size = output_size
    self.rnn = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=stacks)
    self.lastLayer = torch.nn.Linear(hidden_size, output_size)
  
  def forward(self, batch, hidden_state):
    out, next_hidden_state = self.rnn(batch)
    out = self.lastLayer(out)
    return out, next_hidden_state

def test(model, dataPaths):
  softmax = torch.nn.Softmax(dim=2)
  accuracy = 0
  df = pd.DataFrame()
  for path in dataPaths:
    flow = []
    outp = 1
    target = 1 if "Nosynkope" in path else 0
    hidden_state = None
    values = pd.read_csv(path).values
    for sample in values:
      input = torch.from_numpy(sample).view(1, 1, -1)
      pred, hidden_state = model(input, hidden_state)
      pred = softmax(pred)
      pred = pred[-1].view(model.output_size).detach().numpy()
      flow.append(pred[0])
      if pred[0] > 0.95:
        outp = 0
    df[path] = pd.Series(flow)
    if outp == target:
      accuracy += 1
  accuracy /= len(dataPaths)
  return accuracy, df
  
