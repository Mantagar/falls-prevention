from skopt import gp_minimize
import skopt.plots as sp
import matplotlib.pyplot as pp
from model_utils import *
from skopt import load, dump
import sys

def saveCheckpoint(res):
  pass
  
def errorRate(res):
  pass

res = load(sys.argv[1])

print(res.x)
print(res.fun)
print("Iteration no:", len(res.func_vals))

#sp.plot_objective(res)
#pp.show()
