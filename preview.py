import numpy
import os
import matplotlib.pyplot as pp
from pandas import read_csv

loc = "./Processed data/Synkope/"
flist = os.listdir(loc)
for i,f in enumerate(flist):
  print(i,f)
  
while True:
  value = int(input("File index:"))
  data = read_csv(loc+flist[value])

  pp.subplot(2, 1, 1)
  pp.plot(data[['mBP']], label="mean blood pressure [beat]")
  pp.ylabel('mBP')

  pp.subplot(2, 1, 2)
  pp.plot(data[['HR']], label="heart rate [beat]")
  pp.ylabel('HR')

  pp.show()