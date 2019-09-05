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
sns.set_palette('Greys_d')
sns.set_context('notebook')

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

#------------------------------------------
# Load df
df = pd.read_csv('df_wp1_clean.csv', index_col = 0)

# Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip() + '1230')
df.set_index(['IDRSSD','date'],inplace=True)

#------------------------------------------
# Split the dataframes on securitizer and non-securitizers
df_sec = df[df.sec_tot > 0]
df_nonsec = df[df.sec_tot == 0]

#------------------------------------------
# Figure 1: Total securitization (B705 + B711; 1-4 Family residential loans + all other)
## Sum per year, drop index level and divide by $1 million
sec_year = df_sec[['RCB705','RCB706','RCB707','RCB708','RCB709','RCB710','RCB711', 'sec_tot']].sum(level = [1,1])
sec_year = sec_year.droplevel(level = 0)
sec_year = sec_year.divide(1e6)

## Prelims
labels = ['Residential','Home Equity','Credit Card','Auto','Other Consumer','Commercial','All other', 'Total']
line_styles = [(0, (1, 1)),(0, (5, 1)),(0, (3,1,1,1,1,1)),(0, (3,1,1,1)) ,':','-.','--', '-']

##plot
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Total Securitization')
ax.set(xlabel='Year', ylabel = 'Amount of securitization (in $ Billion)')
for i in range(sec_year.shape[1]):
    ax.plot(sec_year.iloc[:,i], linestyle = line_styles[i], label = labels[i], color = 'black')
ax.legend()
plt.tight_layout()
plt.show()

fig.savefig('Fig1_total_sec.png')
