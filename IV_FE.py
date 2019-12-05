#------------------------------------------
# IV treatment model for first working paper
# Robustness checks
# Mark van der Plaat
# November 2019 

''' Changed wrt baseline IV:
    1) Changed ROA to NIM
    2) Changed RC7205 to RC7206 (tier 1 capital ratio)'''


 # Import packages
import pandas as pd
import numpy as np
import scipy.stats as sps # used to calculated cdf and pdf

import os
os.chdir(r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char')

# Import method that adds a constant to a df
from statsmodels.tools.tools import add_constant

# Import method that can estimate a pooled probit
from statsmodels.discrete.discrete_model import Probit

# Import method for POLS (also does FE)
from linearmodels import PanelOLS

# Import packages for the Sargan-Hausman test
from linearmodels.iv._utility import annihilate
from linearmodels.utility import WaldTestStatistic

import sys # to use the help functions needed
sys.path.insert(1, r'X:\My Documents\PhD\Coding_docs\Help_functions')

from summary3 import summary_col

#----------------------------------------------
# Load data and add needed variables

## Load df
df = pd.read_csv('df_wp1_clean.csv', index_col = 0)

## Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip() + '1230')
df.set_index(['IDRSSD','date'],inplace=True)
    
## Drop the missings
df.dropna(subset = ['distance'], inplace = True)

## Dummy variable for loan sales
df['dum_ls'] = (df.ls_tot > 1) * 1

## Make variables that determines the group
df['group'] = (df.index.get_level_values(0).isin(df[df.dum_ls == 1].index.get_level_values(0).to_list())) * 1

# Subset the df
df_ls = df[df.group == 1]   

#----------------------------------------------------
# prelims
## Select and setup the variables
### Dependent variable step 1
w = df.dum_ls

### Dependent variables step 2
y_charge = df.net_coffratio_tot_ta
y_allow = df.allowratio_tot_ta

### Independent exogenous variables
x = df[['cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio']]

x_xbar = df[['cd_pur_ta','cd_sold_ta','RC7205','loanratio','roa','depratio','comloanratio']].transform(lambda df: df - df.mean())

#### Change the columns of x_xbar
dict_x_xbar = dict(zip(['cd_pur_ta','cd_sold_ta','RC7205','loanratio','roa','depratio','comloanratio'],\
         [x + '_xbar' for x in ['cd_pur_ta','cd_sold_ta','RC7205','loanratio','roa','depratio','comloanratio']]))
x_xbar.rename(columns = dict_x_xbar, inplace = True)

### Instruments
z = df[['num_branch', 'perc_full_branch', 'STALPBR', 'distance']]
z_alt = df[['num_branch', 'perc_limited_branch', 'STALPBR', 'distance']]

#----------------------------------------------
#----------------------------------------------    
# MODEL 1: FD IV, charge-off rates
#----------------------------------------------
#----------------------------------------------

#----------------------------------------------
# STEP 1: OLS, with clustered standard errors
#----------------------------------------------

# Estimate G_hat
mod1_step1 = PanelOLS(w, pd.concat([x,z],axis = 1),entity_effects = True, time_effects = True)
res1_step1 = mod1_step1.fit(cov_type = 'clustered', cluster_entity = True)
print(res1_step1.summary)
G_hat = res1_step1.fitted_values

# Calculate G_hat_x_xbar
G_hat_x_xbar = x_xbar * G_hat.values

#----------------------------------------------
# STEP 2: OLS, with clustered standard errors
#----------------------------------------------

# setup the indepedent variables
x_mod1a = pd.concat([x,G_hat],axis = 1)
x_mod1b = pd.concat([x,G_hat,G_hat_x_xbar],axis = 1)

mod1a_step2 = PanelOLS(y_charge,x_mod1a,entity_effects = True, time_effects = True)
res1a_step2 = mod1a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res1a_step2.summary)

mod1b_step2 = PanelOLS(y_charge,x_mod1b,entity_effects = True, time_effects = True)
res1b_step2 = mod1b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res1b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------

# Test weak instruments
## F-test
### First calculate the reduced model (without instruments)
def fTestWeakInstruments(y, fitted_full, fitted_reduced, dof = 4):
    ''' Simple F-test to test the strength of instrumental variables'''
    
    # Calculate the SSE and MSE
    sse_full = np.sum([(y.values[i] - fitted_full.values[i][0])**2 for i in range(y.shape[0])])
    sse_reduced =  np.sum([(y.values[i] - fitted_reduced.values[i][0])**2 for i in range(y.shape[0])])
    
    mse_full = (1 / y.shape[0]) * np.sum([(y.values[i] - fitted_full.values[i][0])**2 for i in range(y.shape[0])])
    
    # Calculate the statistic
    f_stat = ((sse_reduced - sse_full)/4) / mse_full
    
    return f_stat

res1_step1b = PanelOLS(w, x, entity_effects = True, time_effects = True).fit(cov_type = 'clustered', cluster_entity = True)
f_test_mod1 = fTestWeakInstruments(w, res1_step1.fitted_values, res1_step1b.fitted_values, 4) #3239.446832414818

# Test for endogeneity
mod1a_endo = PanelOLS(y_charge,pd.concat([x,w,res1_step1.resids],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod1a_endo.summary) #p-value = 0.0070

# Test for overidentifying restrictions
def sargan(resids, x, z, nendog = 1):

    nobs, ninstr = z.shape
    name = 'Sargan\'s test of overidentification'

    eps = resids.values[:,None]
    u = annihilate(eps, pd.concat([x,z],axis = 1))
    stat = nobs * (1 - (u.T @ u) / (eps.T @ eps)).squeeze()
    null = 'The overidentification restrictions are valid'

    return WaldTestStatistic(stat, null, ninstr - nendog, name=name)

oir_mod1a = sargan(res1a_step2.resids, x, z)
oir_mod1b = sargan(res1b_step2.resids, x, z)
'''NOTE: Both tests are insignificant'''
