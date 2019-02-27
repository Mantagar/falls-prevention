import numpy
import os
import matplotlib.pyplot as pp
from pandas import read_csv
import pandas as pd
import sys

path = sys.argv[1]

data = read_csv(path)

threshold = 0.3
density = 0.001

x = []
y = []
if len(sys.argv)>2:
  threshold = float(sys.argv[2])
  density = 1#only one run of the loop
  
best_accuracy = 0
best_threshold = 0
while threshold<1:
  TP = 0;
  FP = 0;
  TN = 0;
  FN = 0;
  for c in data:
    real_negative = 'Nosynkope' in c
    max_value = data[c].max()
    
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
  pp.gca().legend(("Accuracy","Sensitivity"))#, "Specificity", "PPV (Precision)", "NPV"))
  pp.xlabel('Threshold')
  pp.show()