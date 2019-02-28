import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

csv_files = glob.glob('*.csv')

# Read all prediction files and concatenate them in a single dataframe
dataframes = pd.read_csv(csv_files[0]).add_prefix(csv_files[0].split('/')[-1].split('.')[0])
for i in range(1,len(csv_files)):
    dataframes = pd.concat((dataframes,pd.read_csv(csv_files[i]).drop(columns=['image_name']).add_prefix(csv_files[i].split('/')[-1].split('.')[0])),axis=1)

pred = pd.DataFrame(np.zeros((len(dataframes),5)),columns=['image_name','x1','x2','y1','y2'])
pred.iloc[:,0] = dataframes.iloc[:,0]

# Get model having maximum confidence for each image
m = dataframes.iloc[:,[5*i for i in range(1,len(csv_files)+1)]].idxmax(axis=1)

# For each image select prediction with maximum confidence
for i in tqdm(range(len(dataframes))):
    pred.iloc[i,1:] = dataframes.loc[i,m[i][:12]+'x1':m[i][:12]+'y2'].values

pred.to_csv('../Final_Output.csv',index=False)