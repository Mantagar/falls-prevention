import numpy
import os
import scipy.io as sio
import pandas as pd

def combineMatArrays(split):
  combined = None
  for element in split[0][0]:
    combined = element[0][0] if combined is None else numpy.concatenate([combined, element[0][0]])
  return combined

def processMatlabFile(matfile):
  #filtering .mat file 
  hr = combineMatArrays(matfile['BeatToBeat']['HR'])
  bp = combineMatArrays(matfile['BeatToBeat']['mBP'])
  mapping = {'HR': hr, 'mBP': bp}
  #interpolating data to avoid NaN values
  dataframe = pd.DataFrame(data=mapping)
  dataframe.interpolate(method="linear", inplace=True)
  dataframe.fillna(method="bfill", inplace=True)
  dataframe.fillna(method="ffill", inplace=True)
  #normalization?
  #padding and cutting data to length of 3000
  return dataframe
  
def convertAll(input_folder, output_folder):
  for filename in os.listdir(input_folder):
    print(input_folder+filename+"\t\t----->\t\t"+output_folder+filename)
    matfile = sio.loadmat(input_folder+filename)
    
    dataframe = processMatlabFile(matfile)
    
    filename = filename.replace('.mat','.csv')
    dataframe.to_csv(output_folder+filename, index=False)

convertAll('./Mat/Synkope/','./Processed data/Synkope/')
convertAll('./Mat/No finding/','./Processed data/Nosynkope/')
