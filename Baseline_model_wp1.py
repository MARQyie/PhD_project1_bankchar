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

import sys # to use the help functions needed
sys.path.insert(1, r'X:\My Documents\PhD\Coding_docs\Help_functions')

from summary3 import summary_col

# Import method that adds a constant to a df
from statsmodels.tools.tools import add_constant

#------------------------------------------
# Load df
df = pd.read_csv('df_wp1_clean.csv', index_col = 0)

# Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip() + '1230')
df.set_index(['IDRSSD','date'],inplace=True)

#------------------------------------------
# Dummy variable for loan sales
df['dum_ls'] = (df.ls_tot > 1) * 1

# Make FD variables and drop the missings
y_fd_charge = df.net_coffratio_tot_ta.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
y_fd_allow = df.allowratio_tot_ta.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

x_fd = add_constant(df[['ls_tot_ta','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio']].\
           groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna())
           
x_fd_dum = add_constant(df[['dum_ls','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio']].\
           groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna())
           
# Make a lagged dependent variable and drop the missings
df['net_coffratio_tot_ta_l1'] = df.groupby(df.index.get_level_values(0)).net_coffratio_tot_ta.shift(1)
df['allowratio_tot_ta_l1'] = df.groupby(df.index.get_level_values(0)).allowratio_tot_ta.shift(1)
df.dropna(subset = ['net_coffratio_tot_ta_l1','allowratio_tot_ta_l1'], inplace = True)

# Select and setup the variables
## Dependent variables
y_charge = df.net_coffratio_tot_ta
y_allow = df.allowratio_tot_ta

## Independent variables
x = add_constant(df[['constant','ls_tot_ta','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio']])
x_charge = add_constant(df[['ls_tot_ta','net_coffratio_tot_ta_l1','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio']])

x_allow = add_constant(df[['ls_tot_ta','allowratio_tot_ta_l1','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio']])

#-------------------------------------------
# Analyses
## Pooled with clustered standard errors, with and without dynamic term
model_fe_charge1 = PanelOLS(y_charge,x,entity_effects = True, time_effects = True)
results_fe_charge1 = model_fe_charge1.fit(cov_type = 'clustered', cluster_entity = True)

model_fe_charge2 = PanelOLS(y_charge,x_charge,entity_effects = True, time_effects = True)
results_fe_charge2 = model_fe_charge2.fit(cov_type = 'clustered', cluster_entity = True)

model_fe_allow1 = PanelOLS(y_allow,x,entity_effects = True, time_effects = True)
results_fe_allow1 = model_fe_allow1.fit(cov_type = 'clustered', cluster_entity = True)

model_fe_allow2 = PanelOLS(y_allow,x_allow,entity_effects = True, time_effects = True)
results_fe_allow2 = model_fe_allow2.fit(cov_type = 'clustered', cluster_entity = True)

#-----------------------------------------------
# FE with ls dummy
## Analysis
x = df[['constant','dum_ls','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio']]
x_dum_charge = df[['constant','dum_ls','net_coffratio_tot_ta_l1','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio']]

x_dum_allow = df[['constant','dum_ls','allowratio_tot_ta_l1','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio']]

model_dum_charge1 = PanelOLS(y_charge,x,entity_effects = True, time_effects = True)
results_dum_charge1 = model_dum_charge1.fit(cov_type = 'clustered', cluster_entity = True)

model_dum_charge2 = PanelOLS(y_charge,x_dum_charge,entity_effects = True, time_effects = True)
results_dum_charge2 = model_dum_charge2.fit(cov_type = 'clustered', cluster_entity = True)

model_dum_allow1 = PanelOLS(y_allow,x,entity_effects = True, time_effects = True)
results_dum_allow1 = model_dum_allow1.fit(cov_type = 'clustered', cluster_entity = True)

model_dum_allow2 = PanelOLS(y_allow,x_dum_allow,entity_effects = True, time_effects = True)
results_dum_allow2 = model_dum_allow2.fit(cov_type = 'clustered', cluster_entity = True)

#-----------------------------------------------
# FD models, no dynamic terms
model_fd_charge1 = PanelOLS(y_fd_charge,x_fd)
results_fd_charge1 = model_fd_charge1.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_charge1)

model_fd_charge2 = PanelOLS(y_fd_charge,x_fd_dum)
results_fd_charge2 = model_fd_charge2.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_charge2)

model_fd_allow1 = PanelOLS(y_fd_allow,x_fd)
results_fd_allow1 = model_fd_allow1.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_allow1)

model_fd_allow2 = PanelOLS(y_fd_allow,x_fd_dum)
results_fd_allow2 = model_fd_allow2.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_allow2)
#-------------------------------------------------
# Make table
table1 = summary_col([results_fe_charge1,results_fe_charge2,results_dum_charge1,results_dum_charge2,\
                      results_fe_allow1,results_fe_allow2,results_dum_allow1,results_dum_allow2],\
                     show='se', title='Baseline Model',\
                     regressor_order=['ls_tot_ta','dum_ls','net_coffratio_tot_ta_l1','allowratio_tot_ta_l1'])

table1.to_excel('table_baseline.xlsx')

table2 = summary_col([results_fd_charge1, results_fd_charge2, results_fd_allow1, results_fd_allow2],\
                     show='se', title='Baseline Model - FD',\
                     regressor_order=['ls_tot_ta','dum_ls'])

table2.to_excel('table_baseline_fd.xlsx')