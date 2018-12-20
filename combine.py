from pandas import read_csv
import pandas as pd
import sys

paths = sys.argv[1:]

df = pd.DataFrame()
for name in paths:
  df[name] = read_csv(name, header=None, names=[name])[name]
  
df.to_csv("combined.csv", index=False)
  
