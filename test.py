from model_utils import *
import sys

def check(model, dataPaths, padLeft, padRight):
  accuracy = 0
  for path in dataPaths:
    values = pd.read_csv(path).values
    if len(values)>padLeft:
      values = values[padLeft:]
    if len(values)>padRight:
      values = values[:len(values)-padRight]
    input = torch.from_numpy(values).view(len(values), 1, -1)
    target = 1 if "Nosynkope" in path else 0
    pred, _ = model(input, None)
    pred = pred[-1].view(model.output_size).detach()
    if pred[0]>0.5:
      outp=0
    else:
      outp=1
    if outp==target:
      accuracy += 1
    #print(str(outp)+"("+str(target)+") "+str(pred), flush=True)
  accuracy /= len(dataPaths)
  print(str(padLeft)+'\t'+str(padRight)+'\t'+str(accuracy), flush=True)
  
testDataPaths = loadListFromFile("Processed data/test_set.txt")

model = torch.load(sys.argv[1])

check(model, testDataPaths, 0, 400)

