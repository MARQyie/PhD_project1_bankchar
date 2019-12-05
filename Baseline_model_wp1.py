#------------------------------------------
# Baseline model for first working paper
# Mark van der Plaat
# October 2019 

   
#------------------------------------------
# Import packages
import pandas as pd
import numpy as np
import scipy as sp
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

# Drop missings on distance
df.dropna(subset = ['distance'], inplace = True)

# Dummy variable for loan sales
dum_ls = (df.ls_tot > 1) * 1

# Make variables that determines the group
df['group'] = (df.index.get_level_values(0).isin(df[dum_ls == 1].index.get_level_values(0).to_list())) * 1

# Subset the df
df_ls = df[df.group == 1]   

## Take logs
df = df.select_dtypes(include = ['float64','int']).transform(lambda df: np.log(1 + df)).replace([np.inf, -np.inf], 0)
df_ls = df_ls.select_dtypes(include = ['float64','int']).transform(lambda df: np.log(1 + df)).replace([np.inf, -np.inf], 0)

## Drop NaNs on subset
df.dropna(subset = ['provratio','rwata','net_coffratio_tot_ta','allowratio_tot_ta','ls_tot_ta','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170'], inplace = True)
df_ls.dropna(subset = ['provratio','rwata','net_coffratio_tot_ta','allowratio_tot_ta','ls_tot_ta','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170'], inplace = True)

## Add the dummy variable to the dfs
df['dum_ls'] = dum_ls
df_ls['dum_ls'] = dum_ls

#------------------------------------------
# Make variables for 1) full sample pooled and FE, 2) full sample FD, 3) Subsample FD

# Make FD variables and drop the missings
y_fd_charge = df.net_coffratio_tot_ta.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df)
y_fd_allow = df.allowratio_tot_ta.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df)
y_fd_rwata = df.rwata.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df)
y_fd_prov = df.provratio.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df)

x_fd = df[['ls_tot_ta','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']].\
           groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df)
           
x_fd_dum = df[['dum_ls','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']].\
           groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df)

###
y_ls_fd_charge = df_ls.net_coffratio_tot_ta.groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df)
y_ls_fd_allow = df_ls.allowratio_tot_ta.groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df)
y_ls_fd_rwata = df_ls.rwata.groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df)
y_ls_fd_prov = df_ls.provratio.groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df)

x_ls_fd = df_ls[['ls_tot_ta','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']].\
           groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df)
           
x_ls_fd_dum = df_ls[['dum_ls','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']].\
           groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df)
           
# Make a lagged dependent variable and drop the missings
df['net_coffratio_tot_ta_l1'] = df.groupby(df.index.get_level_values(0)).net_coffratio_tot_ta.shift(1)
df['allowratio_tot_ta_l1'] = df.groupby(df.index.get_level_values(0)).allowratio_tot_ta.shift(1)
df['rwata_l1'] = df.groupby(df.index.get_level_values(0)).rwata.shift(1)
df.dropna(subset = ['net_coffratio_tot_ta_l1','allowratio_tot_ta_l1','rwata_l1'], inplace = True)

# Select and setup the variables
## Dependent variables
y_charge = df.net_coffratio_tot_ta
y_allow = df.allowratio_tot_ta
y_rwata = df.rwata

## Independent variables
x = add_constant(df[['ls_tot_ta','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']])
x_charge = add_constant(df[['ls_tot_ta','net_coffratio_tot_ta_l1','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']])

x_allow = add_constant(df[['ls_tot_ta','allowratio_tot_ta_l1','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']])

x_rwata = add_constant(df[['ls_tot_ta','rwata_l1','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']])

#-------------------------------------------
# Correlationmatrix
corr_matrix = pd.concat([y_fd_charge,y_fd_allow,y_fd_rwata,x_fd,df[['dum_ls']]],axis = 1).corr(method = 'spearman')

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

model_fe_rwata1 = PanelOLS(y_rwata,x,entity_effects = True, time_effects = True)
results_fe_rwata1 = model_fe_rwata1.fit(cov_type = 'clustered', cluster_entity = True)

model_fe_rwata2 = PanelOLS(y_rwata,x_rwata,entity_effects = True, time_effects = True)
results_fe_rwata2 = model_fe_rwata2.fit(cov_type = 'clustered', cluster_entity = True)

#-----------------------------------------------
# FE with ls dummy
## Analysis
x = add_constant(df[['dum_ls','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']])
x_dum_charge = add_constant(df[['dum_ls','net_coffratio_tot_ta_l1','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']])

x_dum_allow = add_constant(df[['dum_ls','allowratio_tot_ta_l1','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']])

x_dum_rwata = add_constant(df[['dum_ls','rwata_l1','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']])

model_dum_charge1 = PanelOLS(y_charge,x,entity_effects = True, time_effects = True)
results_dum_charge1 = model_dum_charge1.fit(cov_type = 'clustered', cluster_entity = True)

model_dum_charge2 = PanelOLS(y_charge,x_dum_charge,entity_effects = True, time_effects = True)
results_dum_charge2 = model_dum_charge2.fit(cov_type = 'clustered', cluster_entity = True)

model_dum_allow1 = PanelOLS(y_allow,x,entity_effects = True, time_effects = True)
results_dum_allow1 = model_dum_allow1.fit(cov_type = 'clustered', cluster_entity = True)

model_dum_allow2 = PanelOLS(y_allow,x_dum_allow,entity_effects = True, time_effects = True)
results_dum_allow2 = model_dum_allow2.fit(cov_type = 'clustered', cluster_entity = True)

model_dum_rwata1 = PanelOLS(y_rwata,x,entity_effects = True, time_effects = True)
results_dum_rwata1 = model_dum_rwata1.fit(cov_type = 'clustered', cluster_entity = True)

model_dum_rwata2 = PanelOLS(y_rwata,x_dum_rwata,entity_effects = True, time_effects = True)
results_dum_rwata2 = model_dum_rwata2.fit(cov_type = 'clustered', cluster_entity = True)

#-----------------------------------------
# Adjusted R-squared
def adjRSqrd(results_panelOLS):
    '''Calculates the Adjusted R-squared from the results of linearmodels.PanelOLS.'''
    # Load the post-estimation results
    rss, tss, n, k = results_panelOLS.resid_ss, results_panelOLS.total_ss,results_panelOLS.nobs, results_panelOLS.params.shape[0]
    
    # Calculate r-squared
    rsqrd = 1 - (rss / tss)
    
    # Calculate adj. r-squared
    adj_rsqrd = 1 - (((1 - rsqrd)*(n - 1))/(n - k - 1))
    
    return adj_rsqrd

adj_rsqrd_fd = []
adj_rsqrd_fd_cd = []

#-----------------------------------------------
# FD models, no dynamic terms
model_fd_charge1 = PanelOLS(y_fd_charge,x_fd)
results_fd_charge1 = model_fd_charge1.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_charge1)
adj_rsqrd_fd.append(adjRSqrd(results_fd_charge1))

''' Is CORRECT
## Test estimates with algebra
yy = y_fd_charge.dropna()
xx = x_fd.dropna()

beta = sp.linalg.inv(xx.T @ xx) @ (xx.T @ yy) #calculate betas
res = np.subtract(yy, xx @ beta) #calculate residuals
sigma = sp.dot(res.T,res) / (len(xx)-1)
'''

model_fd_charge2 = PanelOLS(y_ls_fd_charge,x_ls_fd)
results_fd_charge2 = model_fd_charge2.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_charge2)
adj_rsqrd_fd.append(adjRSqrd(results_fd_charge2))

model_fd_charge3 = PanelOLS(y_fd_charge,x_fd_dum)
results_fd_charge3 = model_fd_charge3.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_charge3)
adj_rsqrd_fd.append(adjRSqrd(results_fd_charge3))

model_fd_charge4 = PanelOLS(y_ls_fd_charge,x_ls_fd_dum)
results_fd_charge4 = model_fd_charge4.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_charge4)
adj_rsqrd_fd.append(adjRSqrd(results_fd_charge4))

model_fd_allow1 = PanelOLS(y_fd_allow,x_fd)
results_fd_allow1 = model_fd_allow1.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_allow1)
adj_rsqrd_fd.append(adjRSqrd(results_fd_allow1))

model_fd_allow2 = PanelOLS(y_ls_fd_allow,x_ls_fd)
results_fd_allow2 = model_fd_allow2.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_allow2)
adj_rsqrd_fd.append(adjRSqrd(results_fd_allow2))

model_fd_allow3 = PanelOLS(y_fd_allow,x_fd_dum)
results_fd_allow3 = model_fd_allow3.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_allow3)
adj_rsqrd_fd.append(adjRSqrd(results_fd_allow3))

model_fd_allow4 = PanelOLS(y_ls_fd_allow,x_ls_fd_dum)
results_fd_allow4 = model_fd_allow4.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_allow4)
adj_rsqrd_fd.append(adjRSqrd(results_fd_allow4))

model_fd_rwata1 = PanelOLS(y_fd_rwata,x_fd)
results_fd_rwata1 = model_fd_rwata1.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_rwata1)
adj_rsqrd_fd.append(adjRSqrd(results_fd_rwata1))

model_fd_rwata2 = PanelOLS(y_ls_fd_rwata,x_ls_fd)
results_fd_rwata2 = model_fd_rwata2.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_rwata2)
adj_rsqrd_fd.append(adjRSqrd(results_fd_rwata2))

model_fd_rwata3 = PanelOLS(y_fd_rwata,x_fd_dum)
results_fd_rwata3 = model_fd_rwata3.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_rwata3)
adj_rsqrd_fd.append(adjRSqrd(results_fd_rwata3))

model_fd_rwata4 = PanelOLS(y_ls_fd_rwata,x_ls_fd_dum)
results_fd_rwata4 = model_fd_rwata4.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_rwata4)
adj_rsqrd_fd.append(adjRSqrd(results_fd_rwata4))

model_fd_prov1 = PanelOLS(y_fd_prov,x_fd)
results_fd_prov1 = model_fd_prov1.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_prov1)
adj_rsqrd_fd.append(adjRSqrd(results_fd_prov1))

model_fd_prov2 = PanelOLS(y_ls_fd_prov,x_ls_fd)
results_fd_prov2 = model_fd_prov2.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_prov2)
adj_rsqrd_fd.append(adjRSqrd(results_fd_prov2))

model_fd_prov3 = PanelOLS(y_fd_prov,x_fd_dum)
results_fd_prov3 = model_fd_prov3.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_prov3)
adj_rsqrd_fd.append(adjRSqrd(results_fd_prov3))

model_fd_prov4 = PanelOLS(y_ls_fd_prov,x_ls_fd_dum)
results_fd_prov4 = model_fd_prov4.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_prov4)
adj_rsqrd_fd.append(adjRSqrd(results_fd_prov4))

#-------------------------------------------------
# Make table
table1 = summary_col([results_fe_charge1,results_fe_charge2,results_dum_charge1,results_dum_charge2,\
                      results_fe_allow1,results_fe_allow2,results_dum_allow1,results_dum_allow2,\
                      results_fe_rwata1,results_fe_rwata2,results_dum_rwata1,results_dum_rwata2],\
                     show='se', title='Baseline Model',\
                     regressor_order=['ls_tot_ta','dum_ls','net_coffratio_tot_ta_l1',\
                                      'allowratio_tot_ta_l1','rwata_l1'])

table1.to_excel('table_baseline.xlsx')

table2 = summary_col([results_fd_charge1, results_fd_charge2, results_fd_charge3, results_fd_charge4,\
                      results_fd_allow1, results_fd_allow2,results_fd_allow3, results_fd_allow4,\
                      results_fd_rwata1, results_fd_rwata2,results_fd_rwata3, results_fd_rwata4,\
                      results_fd_prov1, results_fd_prov2,results_fd_prov3, results_fd_prov4],\
                     show='se', title='Baseline Model - FD',\
                     regressor_order=['ls_tot_ta','dum_ls'])

table2.to_excel('table_baseline_fd.xlsx')

#---------------------------------------------------
# FD models, no dynamic terms without CDs
x_fd.drop(['cd_pur_ta','cd_sold_ta'], axis = 1, inplace = True)
x_ls_fd.drop(['cd_pur_ta','cd_sold_ta'], axis = 1, inplace = True)
x_fd_dum.drop(['cd_pur_ta','cd_sold_ta'], axis = 1, inplace = True)
x_ls_fd_dum.drop(['cd_pur_ta','cd_sold_ta'], axis = 1, inplace = True)

model_fd_charge1 = PanelOLS(y_fd_charge,x_fd)
results_fd_charge1 = model_fd_charge1.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_charge1)
adj_rsqrd_fd_cd.append(adjRSqrd(results_fd_charge1))

model_fd_charge2 = PanelOLS(y_ls_fd_charge,x_ls_fd)
results_fd_charge2 = model_fd_charge2.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_charge2)
adj_rsqrd_fd_cd.append(adjRSqrd(results_fd_charge2))

model_fd_charge3 = PanelOLS(y_fd_charge,x_fd_dum)
results_fd_charge3 = model_fd_charge3.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_charge3)
adj_rsqrd_fd_cd.append(adjRSqrd(results_fd_charge3))

model_fd_charge4 = PanelOLS(y_ls_fd_charge,x_ls_fd_dum)
results_fd_charge4 = model_fd_charge4.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_charge4)
adj_rsqrd_fd_cd.append(adjRSqrd(results_fd_charge4))

model_fd_allow1 = PanelOLS(y_fd_allow,x_fd)
results_fd_allow1 = model_fd_allow1.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_allow1)
adj_rsqrd_fd_cd.append(adjRSqrd(results_fd_allow1))

model_fd_allow2 = PanelOLS(y_ls_fd_allow,x_ls_fd)
results_fd_allow2 = model_fd_allow2.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_allow2)
adj_rsqrd_fd_cd.append(adjRSqrd(results_fd_allow2))

model_fd_allow3 = PanelOLS(y_fd_allow,x_fd_dum)
results_fd_allow3 = model_fd_allow3.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_allow3)
adj_rsqrd_fd_cd.append(adjRSqrd(results_fd_allow3))

model_fd_allow4 = PanelOLS(y_ls_fd_allow,x_ls_fd_dum)
results_fd_allow4 = model_fd_allow4.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_allow4)
adj_rsqrd_fd_cd.append(adjRSqrd(results_fd_allow4))

model_fd_rwata1 = PanelOLS(y_fd_rwata,x_fd)
results_fd_rwata1 = model_fd_rwata1.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_rwata1)
adj_rsqrd_fd_cd.append(adjRSqrd(results_fd_rwata1))

model_fd_rwata2 = PanelOLS(y_ls_fd_rwata,x_ls_fd)
results_fd_rwata2 = model_fd_rwata2.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_rwata2)
adj_rsqrd_fd_cd.append(adjRSqrd(results_fd_rwata2))

model_fd_rwata3 = PanelOLS(y_fd_rwata,x_fd_dum)
results_fd_rwata3 = model_fd_rwata3.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_rwata3)
adj_rsqrd_fd_cd.append(adjRSqrd(results_fd_rwata3))

model_fd_rwata4 = PanelOLS(y_ls_fd_rwata,x_ls_fd_dum)
results_fd_rwata4 = model_fd_rwata4.fit(cov_type = 'clustered', cluster_entity = True)
print(results_fd_rwata4)
adj_rsqrd_fd_cd.append(adjRSqrd(results_fd_rwata4))


table3 = summary_col([results_fd_charge1, results_fd_charge2, results_fd_charge3, results_fd_charge4,\
                      results_fd_allow1, results_fd_allow2,results_fd_allow3, results_fd_allow4,\
                      results_fd_rwata1, results_fd_rwata2,results_fd_rwata3, results_fd_rwata4],\
                     show='se', title='Baseline Model - FD',\
                     regressor_order=['ls_tot_ta','dum_ls'])

table3.to_excel('table_baseline_fd_nocd.xlsx')