import pandas as pd
import sys

paths = sys.argv[1:]

apart = []
for name in paths:
  apart.append(pd.read_csv(name))
df = pd.concat(apart,axis=1)
  
df.to_csv("combined.csv", index=False)