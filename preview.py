import numpy
import os
import matplotlib.pyplot as pp
from pandas import read_csv

loc_s = "./Processed data/Synkope/"
flist_s = os.listdir(loc_s)
loc_n = "./Processed data/Nosynkope/"
flist_n = os.listdir(loc_n)
flist = flist_s + flist_n
for i,f in enumerate(flist):
  print(i,f)
  
while True:
  value = int(input("File index:"))
  if value<len(flist_s):
    data = read_csv(loc_s+flist[value])
  else:
    data = read_csv(loc_n+flist[value])

  pp.subplot(2, 1, 1)
  pp.plot(data[['mBP']], label="mean blood pressure [beat]")
  pp.ylabel('mBP')

  pp.subplot(2, 1, 2)
  pp.plot(data[['HR']], label="heart rate [beat]")
  pp.ylabel('HR')

  pp.show()