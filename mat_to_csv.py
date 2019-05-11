import numpy
import os
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as pp
import sys

def preparePlot(savePlot, df, title, index):
  if savePlot:
    pp.subplot(2, 4, index)
    pp.plot(df[['mBP']])
    pp.ylabel('mBP')
    pp.title(title)

    pp.subplot(2, 4, index+4)
    pp.plot(df[['HR']])
    pp.ylabel('HR')

def combineMatArrays(split):
  combined = None
  for element in split[0][0]:
    combined = element[0][0] if combined is None else numpy.concatenate([combined, element[0][0]])
  return combined

def processMatlabFile(matfile, savePlot, plotFilename):
  #filtering .mat file 
  hr = combineMatArrays(matfile['BeatToBeat']['HR'])
  bp = combineMatArrays(matfile['BeatToBeat']['mBP'])
  mapping = {'HR': hr, 'mBP': bp}
  df = pd.DataFrame(data=mapping)
  #croping first 500 samples and last 50
  df = df.iloc[500:-50]
  preparePlot(savePlot, df, "Original", 1)
  #interpolating data to avoid NaN values
  df = df.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")
  preparePlot(savePlot, df, "Interpolated", 2)
  #outliers removal
  repeatTimes = 5
  for i in range(repeatTimes):
    df_copy = df.copy()
    df_copy = (df - df.mean())/df.std()
    for column_name in df_copy:
      column = df_copy[column_name]
      outliers = column.rolling(window=31, center=True).median().fillna(method='bfill').fillna(method='ffill')
      diff = numpy.abs(column - outliers)
      outlier_ids = diff > 2 / (i+1)
      df[column_name][outlier_ids] = numpy.NaN
    df = df.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")
  preparePlot(savePlot, df, "Removed outliers", 3)
  
  #simple normalization (losing std and mean information)
  #normalization  
  #df = ((df - df.min())/(df.max() - df.min()) - 0.5) * 2
  #preparePlot(savePlot, df, "Normalized", 4)
  
  #using range information (synkope) HR(32,156) mBP(19,220)
  #normalization relative to the whole dataset
  df = ((df - (32,19))/((156-32,220-19)) - 0.5) * 2
  preparePlot(savePlot, df, "Rescaled", 4)
    
  if savePlot:
    pp.gcf().set_size_inches(15, 5)
    pp.subplots_adjust(wspace = 0.25)
    pp.savefig("Charts/Preprocessing/"+plotFilename, dpi=180)
    pp.close()
  
  return df
  
def convertAll(input_folder, output_folder, savePlots):
  count = 0
  maxBP = 0
  minBP = 999
  maxHR = 0
  minHR = 999
  for filename in os.listdir(input_folder):
    matfile = sio.loadmat(input_folder+filename)
    raw_filename = filename.replace('.mat','')
    output_filename = raw_filename + '.csv'
    label = 'Nosynkope' if 'Nosynkope' in output_folder else 'Synkope'
    raw_filename = label + "_" + raw_filename
    df = processMatlabFile(matfile, (count==60 or count==61) and savePlots, raw_filename)
    if df.isnull().values.any()==False and df.shape[0]>500:
      print(input_folder+filename+"\t\t----->\t\t"+output_folder+output_filename,df.shape)    
      df.to_csv(output_folder+output_filename, index=False)
      maxi = df.max()
      mini = df.min()
      if maxi['mBP']>maxBP:
        maxBP = maxi['mBP']
      if mini['mBP']<minBP:
        minBP = mini['mBP']
      if maxi['HR']>maxHR:
        maxHR = maxi['HR']
      if mini['HR']<minHR:
        minHR = mini['HR']
    else:
      print(input_folder+filename+"\t\t----->\t\twas rejected!")
    count += 1
  print("HR("+str(minHR)+";"+str(maxHR)+")")
  print("mBP("+str(minBP)+";"+str(maxBP)+")")

convertAll('./Mat/Synkope/','./data/Synkope/', False)
convertAll('./Mat/No finding/','./data/Nosynkope/', False)
