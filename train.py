from model_utils import *
import sys

trainingDataPaths = loadListFromFile("Processed data/training_set.txt")
validationDataPaths = loadListFromFile("Processed data/validation_set.txt")

input_size = pd.read_csv(trainingDataPaths[0]).shape[1]
hidden_size = 100
stacks = 1
output_size = 2
model = RNN(input_size, hidden_size, stacks, output_size).double()

epochs = 200
batch_size = 20
seq_size = 50
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1/batch_size
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

batcher = Batcher(trainingDataPaths, seq_size, batch_size)

accuracy = 0
#print("EPOCH\t\tTRAINING LOSS\t\t\tVALIDATION LOSS")
for epoch in range(epochs):
  #training
  while batcher.hasNextBatch():
    x, y = batcher.nextBatch()

    optimizer.zero_grad()
    
    pred, _ = model(x, None)
    
    pred = pred.view(batch_size*seq_size, output_size)
    y = y.view(batch_size*seq_size)
    
    loss = loss_fn(pred, y)
    
    loss.backward()
    
    optimizer.step()
    print(loss.detach().numpy(), flush=True)
  current_accuracy = test(model, validationDataPaths)
  if current_accuracy >= accuracy:
    accuracy = current_accuracy
  else:
    break
  torch.save(model, 'checkpoints/'+str(seq_size)+".model")
  batcher.nextEpoch()
  
  
