import numpy
import os
from pandas import read_csv

def inspect(loc):
  flist = os.listdir(loc)
  amount = len(flist)
  avg_len = 0
  max_len = 0
  min_len = 9999999
  for i in flist:
    data = read_csv(loc+i)
    data_len = len(data)
    avg_len += data_len
    max_len = max(data_len, max_len)
    min_len = min(data_len, min_len)
  avg_len /= amount
  
  print("Series:\t" + loc)
  print("Number of distinct series:", amount)
  print("Average length of data series:", int(avg_len))
  print("Minimum length of data series:", min_len)
  print("Maximum length of data series:", max_len)
  print()
  


loc_s = "./data/Synkope/"
inspect(loc_s)


loc_n = "./data/Nosynkope/"
inspect(loc_n)
