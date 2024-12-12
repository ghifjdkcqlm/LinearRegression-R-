import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from youtub import youtub

rr_df = pd.read_csv('robosheet1.csv')
rr_df.drop(rr_df[rr_df.AGE<1].index, inplace = True)
print(rr_df)