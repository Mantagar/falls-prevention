import numpy
import os
import matplotlib.pyplot as pp
from pandas import read_csv
import pandas as pd
import sys

path = sys.argv[1]
step = int(sys.argv[2])

data = read_csv(path)

avg_df = pd.DataFrame()
for col in data.columns.values:
  new = []
  vals = data[col]
  size = len(vals)
  i = 0
  while i<size:
    new.append(numpy.mean(vals[i:i+1000]))
    i+=step
  avg_df[col] = pd.Series(new)

  
avg_df.plot(kind='line')
pp.xlabel('Steps')
pp.ylabel('Loss (average of '+str(step)+' steps)')
pp.legend(title='Learning rate')
pp.title('\nbatch_size = 100\nseq_size = 50')

pp.show()
