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

density = 0.005
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
max_values = data.max()
for threshold in tqdm(numpy.arange(density,1-density, density)):
  if len(sys.argv)>2:
    threshold = float(sys.argv[2])
  TP = 0;
  FP = 0;
  TN = 0;
  FN = 0;
  time_diff_list = []
  for c in data:
    real_negative = 'Nosynkope' in c
    max_value = max_values[c]
    if not real_negative:
      for iter in range(len(data[c])):
        if data[c][iter]>=threshold:
          data_id = c[-8:-4]
          matfile = sio.loadmat('./Mat/Synkope/'+str(data_id)+'.mat')
          time_data = combineMatArrays(matfile['BeatToBeat']['Time'])
          time_diff = time_data[iter + 800] - timestamps[timestamps['data_id'] == data_id]['seconds']
          try:
            if len(sys.argv)>2:
              print(str(data_id)+"\t\t"+str(time_diff.item()))
            time_diff_list.append(time_diff.item())
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
  
  avg_diff = numpy.mean(time_diff_list)
  med_diff = numpy.median(time_diff_list)
  if len(sys.argv)>2:
    print("AVERAGE\t\t"+str(avg_diff))
    print("MEDIAN\t\t"+str(med_diff))
  else:
    y2.append([avg_diff, med_diff])
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
  pp.gca().legend(("Average", "Median"))
  pp.xlabel('Threshold')
  pp.show()
  
  pp.gca().set_autoscale_on(False)
  pp.scatter(x3, y3)
  pp.title("ROC")
  pp.ylabel("True positive rate")
  pp.xlabel('False positive rate')
  pp.show()