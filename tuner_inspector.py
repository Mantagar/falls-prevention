from skopt import gp_minimize
import skopt.plots as sp
import matplotlib.pyplot as pp
from model_utils import *
from skopt import load, dump
import sys

def errorRate(res):
  pass
  
class CheckpointSaver(object):
  def __init__(self, name):
    self.name = name
    
  def __call__(self, res):
     dump(res, 'checkpoints/'+self.name+'.tuner', store_objective=False)

res = load('checkpoints/'+sys.argv[1]+'.tuner')

print(res.x)
print(res.fun)
print("Iteration no:", len(res.func_vals))
print(len(res.models))

sp.plot_objective(res)
pp.show()
