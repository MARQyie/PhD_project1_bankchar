## Plot the total family loans securitized
#------------------------------------------
# Import packages
import pandas as pd
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(style='white',font_scale=1.5)

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

#------------------------------------------
# Load data
df = pd.read_csv('df_wp1_clean.csv', index_col = 0)

# Set multi index
df.set_index( ['IDRSSD', 'date'], inplace = True)

#------------------------------------------
# Plot
sec_fam = df.RCB705.sum(level = 1)

sec_fam.plot()