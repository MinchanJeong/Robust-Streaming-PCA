import pandas as pd
import numpy as np
import glob
import matplotlib
from matplotlib import pyplot as plt
import tqdm

def frame_filter(path, name='Daily Return'):
    df = pd.read_csv(path,infer_datetime_format=True)
    df[name] = df['Adjusted Close'].pct_change()
    df = df[['Date', name]]
    return df

csv_list = sorted(glob.glob('./datasets/sp500/csv/*'))

daily_return_dict = {}
for i,path in enumerate(csv_list):
    name = path[6:-4]
    df = frame_filter(path, name)
    daily_return_dict[name] = df
    
    if i == 0:
        df_total = df
    else:
        df_total = pd.DataFrame.merge(df_total,df,on='Date',how='outer')
        
df_total['Date'] = pd.to_datetime(df_total['Date'],format='%d-%m-%Y').dt.strftime('%Y%m%d')
df_total = df_total.sort_values('Date').reset_index(drop=True)

ii, ff = df_total.index[df_total['Date'] == '19800318'].tolist()[0], df_total.index[df_total['Date'] == '20220722'].tolist()[0]
df_total2 = df_total[ii:ff].dropna(axis=1, how='any').reset_index(drop=True)

arr = df_total2.drop(['Date'], axis=1).values.T
print(arr.shape)
np.save('./datasets/sp500',arr)