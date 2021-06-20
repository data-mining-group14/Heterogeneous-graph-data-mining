#%%
vote_path='./vote/'
import pandas as pd
import numpy as np
import os
#%%
files=os.listdir(vote_path)
results=[]
for file in files:
    name,ext=os.path.splitext(file)
    if ext!=".csv" or name=='vote_result':
        continue
    df = pd.read_csv(vote_path+file,dtype=float)
    results.append(df['label'])
#%%
results=np.array(results)
avg=np.mean(results,axis=0)
#%%
df['label']=avg
# %%
df['id']=df['id'].astype(int)
df.to_csv(vote_path+'vote_result.csv', index=False)
# %%
