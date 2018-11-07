import torch
import pandas as pd
import random

def loadListFromFile(path):
  with open(path, 'r') as file:
    list = file.readlines()
  return [ item.replace('\n', '') for item in list ]

def getData(path):
  df = pd.read_csv(path)
  input = torch.from_numpy(df.values).reshape(df.shape[0], 1, df.shape[1])
  if 'Nosynkope' in path:
    return input, torch.zeros(df.shape[0]).reshape(df.shape[0], 1, 1).double()
  else:
    return input, torch.ones(df.shape[0]).reshape(df.shape[0], 1, 1).double()
  
class RNN(torch.nn.Module):
  def __init__(self, input_size, hidden_size, stacks, output_size):
    super(RNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.stacks = stacks
    self.output_size = output_size    
    self.rnn = torch.nn.RNN(input_size, hidden_size, stacks)
    self.lastLayer = torch.nn.Linear(hidden_size, output_size)
  
  def forward(self, input_seq):
    out, hidden_state = self.rnn(input_seq) 
    return self.lastLayer(out)
  

trainingDataPaths = loadListFromFile("Processed data/training_set.txt")
validationDataPaths = loadListFromFile("Processed data/validation_set.txt")

input_size = 2
hidden_size = 200
stacks = 1
output_size = 1
model = RNN(input_size, hidden_size, stacks, output_size).double()


epochs = 10
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
  for sample in trainingDataPaths:
    data, target = getData(sample)
    
    
    optimizer.zero_grad()
    
    pred = model(data)

    loss = loss_fn(pred, target)
    print(loss)
    loss.backward()
    
    optimizer.step()
  # printing scores for the validation set
  #for sample in validationDataPaths:
  random.shuffle(trainingDataPaths)
  


