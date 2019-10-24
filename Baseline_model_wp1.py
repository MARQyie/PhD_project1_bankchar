#------------------------------------------
# Baseline model for first working paper
# Mark van der Plaat
# October 2019 

   
#------------------------------------------
# Import packages
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 1.75, palette = 'Greys_d')

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

from linearmodels import PanelOLS
from linearmodels.panel import compare
from linearmodels.iolib.summary2 import summary_col

#------------------------------------------
# Load df
df = pd.read_csv('df_wp1_clean.csv', index_col = 0)

# Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip() + '1230')
df.set_index(['IDRSSD','date'],inplace=True)

#------------------------------------------
# Make a lagged dependent variable and drop the missings
df['net_coffratio_tot_ta_l1'] = df.groupby(df.index.get_level_values(0)).net_coffratio_tot_ta.shift(1)
df['allowratio_tot_ta_l1'] = df.groupby(df.index.get_level_values(0)).allowratio_tot_ta.shift(1)
df.dropna(subset = ['net_coffratio_tot_ta_l1','allowratio_tot_ta_l1'], inplace = True)

df['constant'] = np.ones(df.shape[0])

# Select and setup the variables
## Dependent variables
y_charge = df.net_coffratio_tot_ta
y_allow = df.allowratio_tot_ta

## Independent variables
x_charge = df[['constant','ls_tot_ta','net_coffratio_tot_ta_l1','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio']]


x_allow = df[['constant','ls_tot_ta','allowratio_tot_ta_l1','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio']]

#-------------------------------------------
# Analyses
## Pooled with clustered standard errors
model_pols_charge = PanelOLS(y_charge,x_charge)
results_pols_charge = model_pols_charge.fit(cov_type = 'clustered', cluster_entity = True)
print(results_pols_charge)

model_fe_charge = PanelOLS(y_charge,x_charge,entity_effects = True, time_effects = True)
results_fe_charge = model_fe_charge.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fe_charge)

model_pols_allow = PanelOLS(y_allow,x_allow)
results_pols_allow = model_pols_allow.fit(cov_type = 'clustered', cluster_entity = True)
print(results_pols_allow)

model_fe_allow = PanelOLS(y_allow,x_allow,entity_effects = True, time_effects = True)
results_fe_allow = model_fe_allow.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fe_allow)

# Output to latex
summary_col([results_fe_charge,results_fe_allow], stars = True)


print(compare({'FE - Charge-off Ratio':results_fe_charge,'FE - Allowance Ratio':results_fe_allow}, precision = 'std_errors').summary.as_latex())