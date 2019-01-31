import numpy
import os
import scipy.io as sio
import pandas as pd
import matplotlib as plt

def combineMatArrays(split):
  combined = None
  for element in split[0][0]:
    combined = element[0][0] if combined is None else numpy.concatenate([combined, element[0][0]])
  return combined

def processMatlabFile(matfile, cropback):
  #filtering .mat file 
  hr = combineMatArrays(matfile['BeatToBeat']['HR'])
  bp = combineMatArrays(matfile['BeatToBeat']['mBP'])
  sv = combineMatArrays(matfile['BeatToBeat']['SV'])
  mapping = {'HR': hr, 'mBP': bp, 'SV': sv}
  dataframe = pd.DataFrame(data=mapping)
  #croping first 500 and last 500 samples
  if cropback:
    dataframe = dataframe.iloc[500:-500]
  else:
    dataframe = dataframe.iloc[500:]
  #interpolating data to avoid NaN values
  dataframe = dataframe.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")
  #standarization
  dataframe = (dataframe - dataframe.mean())/dataframe.std()
  #outliers removal
  df_copy = dataframe.copy()
  for column in df_copy:
    column = df_copy[column]
    outliers = column.rolling(window=31, center=True).median().fillna(method='bfill').fillna(method='ffill')
    diff = numpy.abs(column - outliers)
    outlier_ids = diff > 2
    column[outlier_ids] = numpy.NaN
  dataframe = df_copy
  #removing NaN values after outliers removal
  dataframe = dataframe.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")
  #normalization
  dataframe = ((dataframe - dataframe.min())/(dataframe.max() - dataframe.min()) - 0.5) * 2
  return dataframe
  
def convertAll(input_folder, output_folder, cropback):
  for filename in os.listdir(input_folder):
    matfile = sio.loadmat(input_folder+filename)
    output_filename = filename.replace('.mat','.csv')
    
    dataframe = processMatlabFile(matfile, cropback)
    if dataframe.isnull().values.any()==False and dataframe.shape[0]>500:
      print(input_folder+filename+"\t\t----->\t\t"+output_folder+output_filename,dataframe.shape)    
      dataframe.to_csv(output_folder+output_filename, index=False)
    else:
      print(input_folder+filename+"\t\t----->\t\twas rejected!")

convertAll('./Mat/Synkope/','./Processed data/Synkope/', False)
convertAll('./Mat/No finding/','./Processed data/Nosynkope/', True)
