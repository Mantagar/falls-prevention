import torch
import pandas as pd
import random
import numpy


class Batcher:
  def __init__(self, dataPaths, seq_size, batch_size):
    self.seq_size = seq_size
    self.batch_size = batch_size
    self.meta = []
    self.data = []
    data_index = 0
    for path in dataPaths:
      values = pd.read_csv(path).values
      target = 1 if "Nosynkope" in path else 0
      seq_index = 0
      while seq_index+seq_size<len(values):
        self.meta.append((target, data_index, seq_index))
        seq_index += 1
      self.data.append(values)
      data_index += 1
    self.sample_amount = len(self.meta)
    self.nextEpoch()
        
  def nextEpoch(self):
    self.meta = random.sample(self.meta, len(self.meta))
    self.id = 0
  
  def hasNextBatch(self):
    return self.id + self.batch_size < self.sample_amount
    
  def nextBatch(self):
    (target, data_index, seq_index) = self.meta[self.id]
    batch_x = torch.from_numpy(self.data[data_index][seq_index:seq_index+self.seq_size]).reshape(self.seq_size, 1, -1)
    batch_y = torch.LongTensor([target]).repeat(self.seq_size).reshape(self.seq_size, 1)
    self.id += 1
    for i in range(self.batch_size-1):
      (target, data_index, seq_index) = self.meta[self.id]
      x = torch.from_numpy(self.data[data_index][seq_index:seq_index+self.seq_size]).reshape(self.seq_size, 1, -1)
      y = torch.LongTensor([target]).repeat(self.seq_size).reshape(self.seq_size, 1)
      batch_x = torch.cat((batch_x, x), 1)
      batch_y = torch.cat((batch_y, y), 1)
      self.id += 1
    return batch_x, batch_y

class RNN(torch.nn.Module):
  def __init__(self, input_size, hidden_size, stacks, output_size):
    super(RNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.stacks = stacks
    self.output_size = output_size
    #VANILLA
    self.rnn = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=stacks)
    self.lastLayer = torch.nn.Linear(hidden_size, output_size)
    #BIDIRECTIONAL
    #self.rnn = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=stacks, bidirectional=True)
    #self.lastLayer = torch.nn.Linear(2*hidden_size, output_size)
  
  def forward(self, batch, hidden_state):
    out, next_hidden_state = self.rnn(batch, hidden_state)
    out = self.lastLayer(out)
    return out, next_hidden_state

def testModel(model, dataPaths):
  softmax = torch.nn.Softmax(dim=2)
  df = pd.DataFrame()
  for path in dataPaths:
    flow = []
    hidden_state = None
    values = pd.read_csv(path).values
    counter = 0
    for sample in values:
      input = torch.from_numpy(sample).view(1, 1, -1)
      pred, hidden_state = model(input, hidden_state)
      pred = softmax(pred)
      pred = pred[-1].view(model.output_size).detach().numpy()
      counter += 1
      if counter>300:
        flow.append(pred[0])
    df[path] = pd.Series(flow)
  return df
  
def trainModel(model, batcher, maxEpochs=1, learningRate=1, learningRateDecay=0.9, printLoss=False):
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adadelta(model.parameters(), lr=learningRate)
  decay_lambda = lambda epoch: learningRateDecay ** epoch
  lr_decay = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_lambda)
  lr_decay.step()

  packetSize = batcher.batch_size*batcher.seq_size

  for epoch in range(maxEpochs):
    if printLoss:
      counter = 0
      avg_loss = 0
      log_every = 10
      
    while batcher.hasNextBatch():
      x, y = batcher.nextBatch()

      optimizer.zero_grad()
      
      pred, _ = model(x, None)
      
      pred = pred.view(packetSize, model.output_size)
      y = y.view(packetSize)
      
      loss = loss_fn(pred, y)
      
      loss.backward()
      
      optimizer.step()
      
      if printLoss:
        avg_loss += loss.detach().numpy()
        counter += 1
        if counter%log_every==0:
          avg_loss /= log_every
          print(avg_loss, flush=True)
          avg_loss = 0
          counter = 0
          
    batcher.nextEpoch()
    lr_decay.step()
  for param_group in optimizer.param_groups:
    return param_group['lr']

def loadDataPaths(path):
  with open(path, 'r') as file:
    list = file.readlines()
  list = [item.replace('\n', '') for item in list]
  return list, pd.read_csv(list[0]).shape[1], 2
  
def getAccuracy(data, threshold):
  correct = 0.
  all = 0.
  for column in data:
    real_negative = 'Nosynkope' in column
    max_value = data[column].max()
    if (real_negative==True and max_value<threshold) or (real_negative==False and max_value>=threshold):
      correct += 1
    all += 1
  return correct/all
