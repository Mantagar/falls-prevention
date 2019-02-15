import numpy
import os
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as pp
import sys

def preparePlot(savePlot, df, p_type, title, index):
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

def processMatlabFile(matfile, p_type, savePlot, plotFilename):
  #filtering .mat file 
  hr = combineMatArrays(matfile['BeatToBeat']['HR'])
  bp = combineMatArrays(matfile['BeatToBeat']['mBP'])
  mapping = {'HR': hr, 'mBP': bp}
  df = pd.DataFrame(data=mapping)
  #croping first 500 samples and last 50
  df = df.iloc[500:-50]
  preparePlot(savePlot, df, p_type, "Original", 1)
  #interpolating data to avoid NaN values
  df = df.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")
  preparePlot(savePlot, df, p_type, "Interpolated", 2)
  #outliers removal
  for i in range(5):
    df_copy = df.copy()
    df_copy = (df - df.mean())/df.std()
    for column_name in df_copy:
      column = df_copy[column_name]
      outliers = column.rolling(window=31, center=True).median().fillna(method='bfill').fillna(method='ffill')
      diff = numpy.abs(column - outliers)
      outlier_ids = diff > 2 / (i+1)
      df[column_name][outlier_ids] = numpy.NaN
    df = df.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")
  preparePlot(savePlot, df, p_type, "Removed outliers", 3)
  
  if p_type==1:#simple normalization (losing std and mean information)
    #normalization  
    df = ((df - df.min())/(df.max() - df.min()) - 0.5) * 2
    preparePlot(savePlot, df, p_type, "Normalized", 4)
  
  if p_type==2:#using range information (synkope) HR(32,156) mBP(19,220)
    #normalization relative to the whole dataset
    df = ((df - (32,19))/((156-32,220-19)) - 0.5) * 2
    preparePlot(savePlot, df, p_type, "Rescaled", 4)
    
  if savePlot:
    pp.gcf().set_size_inches(30, 5)
    pp.subplots_adjust(wspace = 0.15)
    pp.savefig("Charts/Preprocessing/"+str(p_type)+"/"+plotFilename, dpi=200)
    pp.close()
  
  return df
  
def convertAll(input_folder, output_folder, p_type, savePlots):
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
    df = processMatlabFile(matfile, p_type, (count%100==0) and savePlots, raw_filename)
    if df.isnull().values.any()==False and df.shape[0]>500:
      print(input_folder+filename+"\t\t----->\t\t"+output_folder+output_filename,df.shape)    
      df.to_csv(output_folder+output_filename, index=False)
      maxi = df.max()
      mini = df.min()
      if maxi['mBP']>maxBP:
        maxBP = maxi['mBP']
      if maxi['HR']>maxHR:
        maxHR = maxi['HR']
      if mini['mBP']<minBP:
        minBP = mini['mBP']
      if mini['HR']<minHR:
        minHR = mini['HR']
    else:
      print(input_folder+filename+"\t\t----->\t\twas rejected!")
    count += 1
  print("HR("+str(minHR)+";"+str(maxHR)+")")
  print("mBP("+str(minBP)+";"+str(maxBP)+")")

p_type = int(sys.argv[1])
convertAll('./Mat/Synkope/','./Processed data/Synkope/', p_type, False)
convertAll('./Mat/No finding/','./Processed data/Nosynkope/', p_type, False)
