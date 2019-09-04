#------------------------------------------
# Make table displays the number of (non-)securitizers per year for first working paper
# Mark van der Plaat
# September 2019 

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
# Make table
## Count rows per year (securitizers vs. nonsecuritizers)
count_sec = df[df.RCBtot > 1].groupby(level = [1,1]).size()
count_nonsec = df[df.RCBtot == 0].groupby(level = [1,1]).size()

### Remove double index
count_sec = count_sec.droplevel(level = 0)
count_nonsec = count_nonsec.droplevel(level = 0)

## Make total column
count_sum = count_sec + count_nonsec

## Make dataframe
table_count = pd.DataFrame([count_sec,count_nonsec,count_sum],\
                           index = ['Securitizers','Non-securitizers','Total']).T

## Make total row                                          
table_count.loc[-1] = np.array(table_count.sum())
table_count.rename({-1:'Total'}, axis = 'index', inplace = True)

## Save table
table_count.to_excel('Table_count_sec_nonsec.xlsx')