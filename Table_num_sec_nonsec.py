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
os.chdir(r'X:\My Documents\PhD\Materials_papers\1_Working_paper_loan_sales')

#------------------------------------------
# Load df
df = pd.read_csv('Data/df_wp1_clean.csv', index_col = 0)

# Make multi index
df.set_index(['IDRSSD','date'],inplace=True)

#------------------------------------------
# Make table
## Count rows per year (banks with securitized loans sales, non-securitized loan sales, both and nothing)
count_lssec = df[df.ls_sec_tot > 0].groupby(level = [1,1]).size()
count_lsnonsec = df[df.ls_nonsec_tot > 0].groupby(level = [1,1]).size()
count_ls = df[df.ls_tot > 0].groupby(level = [1,1]).size()
count_non = df[df.ls_tot == 0].groupby(level = [1,1]).size()

## Make total column
count_sum = df.groupby(level = [1,1]).size().astype(int)

### Remove double index
count_sum = count_sum.droplevel(level = 0)
count_ls = count_ls.droplevel(level = 0)
count_lssec = count_lssec.droplevel(level = 0)
count_non = count_non.droplevel(level = 0)
count_lsnonsec = count_lsnonsec.droplevel(level = 0)

## Make a percentage securitizers column
perc_lssec = count_lssec.divide(count_sum) * 100
prec_ls = count_ls.divide(count_sum) * 100

## Make dataframe
table_count = pd.DataFrame([count_lssec,perc_lssec,count_lsnonsec,count_ls,\
                            prec_ls,count_non,count_sum],\
                           index = ['Securitized Loan Sales','Securitized Loan Sales (in %)',\
                                    'Non-securitized Loan Sales','Total Loan Sales','Total Loan Sales (in %)',\
                                    'No Loan Sales','Total']).T

## Make total row
### First sum all the columns:
sum_col = table_count.sum(axis = 0).astype(int).tolist()

### Recalculate item 1 and 4 
sum_col[1] = sum_col[0] / sum_col[-1] * 100
sum_col[4] = sum_col[3] / sum_col[-1] * 100

## Add to the table
table_count.loc[-1] = np.array(sum_col)

### Rename the last row                            
table_count.rename({-1:'Total Sample'}, axis = 'index', inplace = True)

#----------------------------------------
# Clean up the table
## Round the numnber, % 2 decimals, ints 0 decimals
table_count.iloc[:,[1,4]] = table_count.iloc[:,[1,4]].round(2)
table_count.iloc[:,[0,2,3,5,6]] = table_count.iloc[:,[0,2,3,5,6]].astype(int)

## Reset index
table_count.reset_index(inplace = True)

## Fix the date column
table_count.iloc[:-1,0] = table_count.iloc[:-1,0].str[:4]
table_count.rename(columns = {'date':''},inplace = True)

#----------------------------------------
## Save table
table_count.to_excel('Tables\Table_ls.xlsx', index = False)
table_count.to_latex('Tables\Table_ls_latex.tex')