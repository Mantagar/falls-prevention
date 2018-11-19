from model_utils import *

trainingDataPaths = loadListFromFile("Processed data/training_set.txt")
validationDataPaths = loadListFromFile("Processed data/validation_set.txt")

input_size = pd.read_csv(trainingDataPaths[0]).shape[1]
hidden_size = 200
stacks = 1
output_size = 2
model = RNN(input_size, hidden_size, stacks, output_size).double()

epochs = 10
batch_size = 20
seq_size = 100
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

target_syncope = torch.LongTensor([0]).repeat(batch_size)
target_nosyncope = torch.LongTensor([1]).repeat(batch_size)

#print("EPOCH\t\tTRAINING LOSS\t\t\tVALIDATION LOSS")
for epoch in range(epochs):
  #training
  for path in trainingDataPaths:
    df = pd.read_csv(path)
    input = df.values
    target = target_nosyncope if "Nosynkope" in path else target_syncope
    seq_index = 0
    while seq_index+seq_size+batch_size<len(input):
      #break
      batch, seq_index = createMiniBatch(input, seq_index, seq_size, batch_size, input_size)

      optimizer.zero_grad()
      
      pred = model(batch)
      pred = pred[-1].view(batch_size, output_size)

      loss = loss_fn(pred, target)
      
      loss.backward()
      
      optimizer.step()
      #print(loss.detach().numpy())
    torch.save(model, 'checkpoints/epoch'+str(epoch))
    '''
    batch, _ = createMiniBatch(input, 0, len(input), 1, input_size)
    pred = model(batch)[-1].view(1, output_size)
    print(str(loss_fn(pred, target.view(batch_size,1)[0]).detach().numpy()) + "\t" + ('Nosynokpe' if "Nosynkope" in path else 'Synkope'))
    '''
  random.shuffle(trainingDataPaths)
  
  #testing prediction on validation and trainig set
  #print(str(epoch) + "\t\t" + str(calculateAverageLoss(model, trainingDataPaths)) + "\t\t" + str(calculateAverageLoss(model, validationDataPaths)))
  
  
  