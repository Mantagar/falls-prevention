from model_utils import *
import sys

trainingDataPaths = loadListFromFile("Processed data/training_set.txt")

input_size = pd.read_csv(trainingDataPaths[0]).shape[1]
output_size = 2
hidden_size = 100
stacks = 1
model = RNN(input_size, hidden_size, stacks, output_size).double()

epochs = 200
batch_size = 16
seq_size = 50
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1

optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
decay_lambda = lambda epoch: 0.9 ** epoch
lr_decay = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_lambda)

batcher = Batcher(trainingDataPaths, seq_size, batch_size)

#print("EPOCH\t\tTRAINING LOSS\t\t\tVALIDATION LOSS")
for epoch in range(epochs):
  #training
  lr_decay.step()
  #print(optimizer.param_groups[0]['lr'])
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
  torch.save(model, 'checkpoints/'+str(seq_size)+"_"+str(epoch)+'_'+str(learning_rate)+".model")
  batcher.nextEpoch()
  
  
