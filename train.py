from model_utils import *
import sys

trainingDataPaths = loadListFromFile("Processed data/training_set.txt")
validationDataPaths = loadListFromFile("Processed data/validation_set.txt")

input_size = pd.read_csv(trainingDataPaths[0]).shape[1]
hidden_size = 200
stacks = 1
output_size = 2
model = RNN(input_size, hidden_size, stacks, output_size).double()

epochs = 10
batch_size = 100
seq_size = 50
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = float(sys.argv[1])
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

target_syncope = torch.LongTensor([0]).repeat(batch_size*seq_size)
target_nosyncope = torch.LongTensor([1]).repeat(batch_size*seq_size)
batcher = Batcher(trainingDataPaths, seq_size, batch_size)

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
  torch.save(model, 'checkpoints/'+str(learning_rate)+".model")
  batcher.nextEpoch()
  
  