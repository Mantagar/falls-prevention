import numpy
import os
import matplotlib.pyplot as pp
from pandas import read_csv
import pandas as pd
import sys

path = sys.argv[1]
step = int(sys.argv[2])

data = read_csv(path)

avgdf = pd.DataFrame()
id_max = []
val_max = []
if step==0:
  #Show only maximum obtained value
  for col in data:
    id_max.append(data[col].idxmax())
    val_max.append(data[col][id_max[-1]])
    avgdf[col] = data[col]
else:
  for col in data:
    avgdf[col] = data[col].rolling(step).mean().shift(-step+1)
    
colors = []
for col in avgdf:
  if 'Nosynkope' in col:
    colors.append('red')
  elif 'Synkope' in col:
    colors.append('green')
if step==0:
  pp.scatter(x=id_max, y=val_max, color=colors)
  pp.xlabel('Steps')
  pp.ylabel('Classification (maximum)')
else:
  avgdf.plot(kind='line', color=colors, legend=None)
  pp.xlabel('Steps')
  pp.ylabel('Classification (average of '+str(step)+' steps)')
  #pp.legend(title='Series')
  pp.title('')

pp.show()
