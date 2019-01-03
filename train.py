from model_utils import *
import sys

trainingDataPaths = loadListFromFile("Processed data/training_set.txt")
validationDataPaths = loadListFromFile("Processed data/validation_set.txt")

input_size = pd.read_csv(trainingDataPaths[0]).shape[1]
output_size = 2
hidden_size = int(sys.argv[1])
stacks = 2
model = RNN(input_size, hidden_size, stacks, output_size).double()

epochs = 200
batch_size = 16
seq_size = 300
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1

optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
decay_lambda = lambda epoch: 0.9 ** epoch
lr_decay = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_lambda)

batcher = Batcher(trainingDataPaths, seq_size, batch_size)

for epoch in range(epochs):
  #training
  lr_decay.step()
  counter = 0
  avg_loss = 0
  log_every = 10
  while batcher.hasNextBatch():
    x, y = batcher.nextBatch()

    optimizer.zero_grad()
    
    pred, _ = model(x, None)
    
    pred = pred.view(batch_size*seq_size, output_size)
    y = y.view(batch_size*seq_size)
    
    loss = loss_fn(pred, y)
    
    loss.backward()
    
    optimizer.step()

    avg_loss += loss.detach().numpy()
    counter += 1
    if counter%log_every==0:
      avg_loss /= log_every
      print(avg_loss, flush=True)
      avg_loss = 0
      counter = 0
  accuracy, df = test(model, validationDataPaths)
  name = str(hidden_size)+'_'+str(epoch)+'_'+str(accuracy)
  df.to_csv(name+'.csv', index=False)
  torch.save(model, 'checkpoints/'+name+".model")
  batcher.nextEpoch()
  
  
