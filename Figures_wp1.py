#------------------------------------------
# Make figures for first working papers
# Mark van der Plaat
# June 2019 

''' 
    This script creates for working paper #1
    The tables display the mean, median, standard deviation for the total sample
    securitizers and non-securitizers. 

    Data used: US Call reports 2001-2018 year-end
    
    First run: Data_setup_wp1.py and Clean_data_wp1.py
    ''' 
    
#------------------------------------------
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(style='white',font_scale=1.5)

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

#------------------------------------------
# Load df
df = pd.read_csv('df_wp1_clean.csv', index_col = 0 )

# Make multi index
df.set_index(['IDRSSD','date'],inplace=True)

#------------------------------------------
# Split the dataframes on securitizer and non-securitizers
df_sec = df[df.RCBtot > 0]
df_nonsec = df[df.RCBtot == 0]