import numpy
import os
import matplotlib.pyplot as pp
import scipy.io as sio
from pandas import read_csv
import pandas as pd
import sys
from tqdm import tqdm

def combineMatArrays(split):
  combined = None
  for element in split[0][0]:
    combined = element[0][0] if combined is None else numpy.concatenate([combined, element[0][0]])
  return combined
  
path = sys.argv[1]

data = read_csv(path)

density = 0.01
x = []
y = []
y2 = []
x3 = []
y3 =[]
if len(sys.argv)>2:
  threshold = float(sys.argv[2])
  print("data_id\t\ttime_diff")
  
timestamps = pd.read_csv("Synkope_timestamps.csv")
best_score = 0
best_threshold = 0
for loopVal in tqdm(range(int(density*100),int(100-density*100), int(density*100))):
  threshold = loopVal/100
  if len(sys.argv)>2:
    threshold = float(sys.argv[2])
  TP = 0;
  FP = 0;
  TN = 0;
  FN = 0;
  time_diff_count = 0.00000001
  time_diff_sum = 0
  for c in data:
    real_negative = 'Nosynkope' in c
    max_value = data[c].max()
    if not real_negative:
      for iter in range(len(data[c])):
        if data[c][iter]>=threshold:
          data_id = c[-8:-4]
          matfile = sio.loadmat('./Mat/Synkope/'+str(data_id)+'.mat')
          time_data = combineMatArrays(matfile['BeatToBeat']['Time'])
          time_diff = time_data[iter + 800] - timestamps[timestamps['data_id'] == data_id]['seconds']
          try:
            if density==1:\
              print(str(data_id)+"\t\t"+str(time_diff.item()))
            time_diff_sum += time_diff.item()
            time_diff_count += 1
          except:
            pass
          break
    
    
    if real_negative:
      if max_value>=threshold:#classified as synkope
        FP += 1
      else:
        TN += 1
    else:
      if max_value>=threshold:
        TP += 1
      else:
        FN += 1
  
  e = 0.00000001
  accuracy = (TP + TN) / (TP + TN + FP + FN + e)
  errorRate = 1 - accuracy
  sensitivity = TP / (TP + FN + e)
  specificity = TN / (TN + FP + e)
  PPV = TP / (TP + FP + e)
  NPV = FP / (FP + FN + e)
  f1 = 2*sensitivity*PPV/(sensitivity+PPV+e)
  
  if density==1:
    print("AVG\t\t"+str(time_diff_sum/time_diff_count))
  else:
    y2.append(time_diff_sum/time_diff_count)
    x3.append(1-specificity)
    y3.append(sensitivity)
 
  x.append(threshold)
  y.append([accuracy, f1])
  if f1 > best_score:
    best_score = f1
    best_threshold = threshold
    log = "\n"
    log += "TP = "+str(TP)+"\tFP = "+str(FP)+"\nFN = "+str(FN)+"\tTN = "+str(TN)+"\n"
    log += "errorRate = "+str(errorRate)+"\n"
    log += "sensitivity = "+str(sensitivity)+"\n"
    log += "specificity = "+str(specificity)+"\n"
    log += "PPV = "+str(PPV)+"\n"
    log += "NPV = "+str(NPV)+"\n"
    log += "\n"
    log += "accuracy = "+str(accuracy)+"\n"
    log += "F1 = "+str(f1)+"\n"
    
  if len(sys.argv)>2:
    break 

print(log)
print("Best threshold: "+str(best_threshold))
print()
if len(sys.argv)<=2:
  pp.gca().set_autoscale_on(False)
  pp.plot(x, y)
  pp.gca().legend(("Accuracy", "F1"))
  pp.xlabel('Threshold')
  pp.show()
  
  pp.plot(x, y2)
  pp.title("Reaction time is the time difference\nbetween model's and manual presyncope detection")
  pp.ylabel("Reaction time [s]")
  pp.xlabel('Threshold')
  pp.show()
  
  pp.gca().set_autoscale_on(False)
  pp.scatter(x3, y3)
  pp.title("AUC")
  pp.ylabel("True positive rate")
  pp.xlabel('False positive rate')
  pp.show()