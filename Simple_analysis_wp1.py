#------------------------------------------
# Simple analysis for first working paper
# Mark van der Plaat
# September 2019 

   
#------------------------------------------
# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 1.75, palette = 'Greys_d')

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

import statsmodels.formula.api as smf

#------------------------------------------
# Load df
df = pd.read_csv('df_wp1_clean.csv', index_col = 0)

# Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip() + '1230')
df.set_index(['IDRSSD','date'],inplace=True)

#------------------------------------------
# Choose the variables

## Dependent variables
df['tot_chargeoff'] = df[['RIADB747','RIADB748','RIADB749','RIADB750','RIADB751','RIADB752','RIADB753','RIAD4635']].\
        sum(axis = 1).divide(df.RC2122 + df.ls_tot).replace(np.inf, 0.0)
df['tot_allowance'] = df[['RCONB557','RIAD3123']].sum(axis = 1).divide(df.RC2122 + df.ls_tot).replace(np.inf, 0.0) 

## Independent variables
#x = ' + '.join(['tot_size', 'ls_tot', 'cd_pur', 'cd_sold', 'RC7205', 'roa'])     
x = ' + '.join(['size','ls_tot', 'cd_pur', 'cd_sold'])      

#------------------------------------------
# Analysis
results_pols1 = smf.ols('tot_chargeoff ~ {}'.format(x), data=df).fit()
print(results_pols1.summary())
print(results_pols1.summary().as_latex())

results_pols2 = smf.ols('tot_allowance ~ {}'.format(x), data=df).fit()
print(results_pols2.summary())
print(results_pols2.summary().as_latex())