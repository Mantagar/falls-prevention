import numpy
import os
import matplotlib.pyplot as pp
import scipy.io as sio
from pandas import read_csv
import pandas as pd
import sys

def combineMatArrays(split):
  combined = None
  for element in split[0][0]:
    combined = element[0][0] if combined is None else numpy.concatenate([combined, element[0][0]])
  return combined
  
path = sys.argv[1]

data = read_csv(path)

threshold = 0.3
density = 0.01

x = []
y = []
x2 = []
y2 = []
if len(sys.argv)>2:
  threshold = float(sys.argv[2])
  density = 1#only one run of the loop
  print("data_id\t\ttime_diff")
  
#TODO wykres średniej różnicy czasu od accuracy dla tego i z bayesem
  
timestamps = pd.read_csv("Synkope_timestamps.csv")
best_accuracy = 0
best_threshold = 0
while threshold<1:
  TP = 0;
  FP = 0;
  TN = 0;
  FN = 0;
  time_diff_count = 0
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
  
  if density==1:
    print("AVG\t\t"+str(time_diff_sum/time_diff_count))
  else:
    x2.append(accuracy)
    y2.append(time_diff_sum/time_diff_count)
 
  x.append(threshold)
  y.append([accuracy, sensitivity])#, specificity, PPV, NPV])
  if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_threshold = threshold
    log = "\n"
    log += "TP = "+str(TP)+"\tFP = "+str(FP)+"\nFN = "+str(FN)+"\tTN = "+str(TN)+"\n"
    log += "errorRate = "+str(errorRate)+"\n"
    log += "sensitivity = "+str(sensitivity)+"\n"
    log += "specificity = "+str(specificity)+"\n"
    log += "PPV = "+str(PPV)+"\n"
    log += "NPV = "+str(NPV)+"\n"
  threshold += density

print(log)
print("Best threshold: "+str(best_threshold))
print("Best accuracy: "+str(best_accuracy))
print()
if len(sys.argv)<=2:
  pp.plot(x, y)
  pp.gca().legend(("Accuracy","Sensitivity"))
  pp.xlabel('Threshold')
  pp.show()
  
  pp.plot(x2, y2)
  pp.title("Reaction time is the time difference\nbetween model's and manual presyncope detection")
  pp.ylabel("Reaction time [s]")
  pp.xlabel('Accuracy')
  pp.show()