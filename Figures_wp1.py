#------------------------------------------
# Make figures for first working papers
# Mark van der Plaat
# September 2019 

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
import seaborn as sns
sns.set(style='white',font_scale=1.5)
sns.set_palette('Greys')
sns.set_context('notebook')

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

#------------------------------------------
# Load df
df = pd.read_csv('df_wp1_clean.csv', index_col = 0)

# Make multi index
df.set_index(['IDRSSD','date'],inplace=True)

#------------------------------------------
# Split the dataframes on securitizer and non-securitizers
df_sec = df[df.sec_tot > 0]
df_nonsec = df[df.sec_tot == 0]

#------------------------------------------
# Figure 1: Total securitization (B705 + B711; 1-4 Family residential loans + all other)
## Sum per year, drop index level and divide by $1 million
sec_year = df_sec.sec_tot.mean(level = [1,1])
sec_year = sec_year.droplevel(level = 0)
sec_year = sec_year.divide(1e6)

##plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Securitization of Residential Loans')
ax.set(xlabel='Year', ylabel = 'Amount of securitization (in $ Million)')
ax.plot(sec_year)
