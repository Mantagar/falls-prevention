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
for col in data:
  avgdf[col] = data[col].rolling(step, center=True).mean().fillna(method='ffill').fillna(method='bfill')
  
colors = []
for col in avgdf:
  if 'Nosynkope' in col:
    colors.append('red')
  elif 'Synkope' in col:
    colors.append('green')

if len(colors)==0:
  colors = None
  avgdf.plot(kind='line', color=colors)
  pp.xlabel('Steps')
  pp.ylabel('Loss (average of '+str(step)+' steps)')
  pp.legend(title='Learning rate')
  pp.title('batch_size = 16')
else:
  avgdf.plot(kind='line', color=colors)
  pp.xlabel('Steps')
  pp.ylabel('Classification (average of '+str(step)+' steps)')
  pp.legend(title='Series')
  pp.title('')

pp.show()
