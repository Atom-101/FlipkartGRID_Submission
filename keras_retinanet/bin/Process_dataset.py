import pandas as pd
import numpy as np

df = pd.read_csv('training.csv')

# Remove malformed datapoints
df = df[df['y1']<df['y2']-5]
df = df[df['x1']<df['x2']-5]
# print(len(df))

# Reorder columns for retinanet
df = df.reindex(columns=['image_name','x1','y1','x2','y2'])

df['image_name'] = '/home/atom/common_data/Projects/FlipkartGRID/Dataset/images/'+df['image_name']

# Make a classes column. Everything belongs to same class.
df['classes'] = np.zeros_like(df['x1'].values)

df.to_csv('valid.csv',index=False,header=False)