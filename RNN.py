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

  
class RNN(torch.nn.Module):
  def __init__(self, input_size, hidden_size, stacks, output_size):
    super(RNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.stacks = stacks
    self.output_size = output_size
    self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=stacks, batch_first=True)
    self.lastLayer = torch.nn.Linear(hidden_size, output_size)
    self.activFunc = torch.nn.Sigmoid()
  
  def forward(self, batch):
    out, hidden_state = self.rnn(batch)
    out = self.lastLayer(out)
    return self.activFunc(out)
  

trainingDataPaths = loadListFromFile("Processed data/training_set.txt")
validationDataPaths = loadListFromFile("Processed data/validation_set.txt")

input_size = pd.read_csv(trainingDataPaths[0]).shape[1]
hidden_size = 100
stacks = 2
output_size = 1
model = RNN(input_size, hidden_size, stacks, output_size).double()

epochs = 10
batch_size = 20
seq_size = 100
loss_fn = torch.nn.MSELoss(reduction="sum")
learning_rate = 0.001
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
  #training
  for path in trainingDataPaths:
    #do i want to read every time or maybe cache it in the future?
    df = pd.read_csv(path)
    input = df.values
    target = torch.zeros(seq_size, batch_size, 1).double() if "Nosynkope" in path else torch.ones(seq_size, batch_size, 1).double()
    seq_index = 0
    while seq_index+seq_size+batch_size<len(input):
      batch, seq_index = createMiniBatch(input, seq_index, seq_size, batch_size, input_size)

      optimizer.zero_grad()
      
      pred = model(batch)
      
      loss = loss_fn(pred, target)
      
      loss.backward()
      
      optimizer.step()
    print(loss.detach().numpy())
  random.shuffle(trainingDataPaths)
  
  #testing prediction on validation set
  print("VALIDATION")
  sample_seq_sizes = [500, 1000]
  for path in validationDataPaths:
    df = pd.read_csv(path)
    input = df.values
    target = 0 if "Nosynkope" in path else 1
    raport = ""
    for sample_seq_size in sample_seq_sizes:
      batch, _ = createMiniBatch(input, 0, sample_seq_size, 1, input_size)
      
      pred = model(batch)
      pred = pred[-1,0,:].detach().numpy()
      raport += str(sample_seq_size) + ": " + str(pred) + "\t"
    raport += "EXPECTED: " + str(target)
    print(raport)


