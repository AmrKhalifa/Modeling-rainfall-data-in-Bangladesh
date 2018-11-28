import pandas as pd 
import numpy as np 



x = np.array([1,2,3]).reshape(1,-1)
df = pd.DataFrame(x)

print(df.head())