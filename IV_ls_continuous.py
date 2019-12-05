#------------------------------------------
# IV treatment model for first working paper
# Mark van der Plaat
# October 2019 

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

#--------------------------------------------
''' This script estimates the treatment effect of loan sales on credit risk
    with an IV estimation procedure. The procedure has two steps
    
    Step 1: Estimate a probit I(LS > 0) on 1, X and Z, where X are the exogenous
    variables and Z are the instruments. Obtain the fitted probabilities G()
    
    Step 2: Do a OLS of CR on 1, G_hat, X, G_hat(X-X_bar)
    
    The first model does not explicitely correct for fixed effects.
    
    Dynamic effects are not included.
    '''   

# Test for overidentifying restrictions
def sargan(resids, x, z, nendog = 1):

    nobs, ninstr = z.shape
    name = 'Sargan\'s test of overidentification'

    eps = resids.values[:,None]
    u = annihilate(eps, pd.concat([x,z],axis = 1))
    stat = nobs * (1 - (u.T @ u) / (eps.T @ eps)).squeeze()
    null = 'The overidentification restrictions are valid'

    return WaldTestStatistic(stat, null, ninstr - nendog, name=name)

#----------------------------------------------
# Load data and add needed variables

# Load df
df = pd.read_csv('df_wp1_clean.csv', index_col = 0)

# Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip() + '1230')
df.set_index(['IDRSSD','date'],inplace=True)

# Drop missings on distance
df.dropna(subset = ['distance'], inplace = True)

# Dummy variable for loan sales
dum_ls = (df.ls_tot > 0) * 1

# Make variables that determines the group
df['group'] = (df.index.get_level_values(0).isin(df[dum_ls == 1].index.get_level_values(0).to_list())) * 1

# Subset the df
#df_ls = df[df.group == 1]

## Kick out the community banks (based on Stiroh, 2004)   
#df_ls = df[~((df.RC2170 < 3e5) & (df.bhc == 0))]

## Only take the banks that change in dum_ls 
intersect = np.intersect1d(df[df.ls_tot > 0].index.get_level_values(0).unique(), df[df.ls_tot == 0].index.get_level_values(0).unique())
df_ls = df[df.index.get_level_values(0).isin(intersect)]
dum_ls_ls = (df_ls.ls_tot > 0) * 1

## Take logs
df = df.select_dtypes(include = ['float64','int']).transform(lambda df: np.log(1 + df)).replace([np.inf, -np.inf], 0)
df_ls = df_ls.select_dtypes(include = ['float64','int']).transform(lambda df: np.log(1 + df)).replace([np.inf, -np.inf], 0)

## Drop NaNs on subset
df.dropna(subset = ['rwata','net_coffratio_tot_ta','allowratio_tot_ta','ls_tot_ta','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170'], inplace = True)
df_ls.dropna(subset = ['rwata','net_coffratio_tot_ta','allowratio_tot_ta','ls_tot_ta','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170'], inplace = True)

## Add the dummy variable to the dfs
df['dum_ls'] = dum_ls
df_ls['dum_ls'] = dum_ls_ls

#----------------------------------------------
# Prelims

## Select and setup the variables
### Dependent variable step 1
w_fd = df.ls_tot_ta.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
w_ls_fd = df_ls.ls_tot_ta.groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

### Dependent variables step 2
y_charge_fd = df.net_coffratio_tot_ta.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
y_charge_ls_fd = df_ls.net_coffratio_tot_ta.groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

y_allow_fd = df.allowratio_tot_ta.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
y_allow_ls_fd = df_ls.allowratio_tot_ta.groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

### Independent exogenous variables
x_fd = df[['cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']].groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
x_ls_fd = df_ls[['cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']].groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

x_xbar_fd = df[['cd_pur_ta','cd_sold_ta','RC7205','loanratio','roa','depratio','comloanratio','RC2170']].transform(lambda df: df - df.mean()).groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
x_xbar_ls_fd = df_ls[['cd_pur_ta','cd_sold_ta','RC7205','loanratio','roa','depratio','comloanratio','RC2170']].transform(lambda df: df - df.mean()).groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

#### Change the columns of x_xbar
dict_x_xbar = dict(zip(['cd_pur_ta','cd_sold_ta','RC7205','loanratio','roa','depratio','comloanratio','RC2170'],\
         [x + '_xbar' for x in ['cd_pur_ta','cd_sold_ta','RC7205','loanratio','roa','depratio','comloanratio','RC2170']]))
x_xbar_fd.rename(columns = dict_x_xbar, inplace = True)
x_xbar_ls_fd.rename(columns = dict_x_xbar, inplace = True)

### Instruments
#z_fd = df[['num_branch', 'STALPBR']].groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
#z_ls_fd = df_ls[['num_branch', 'STALPBR']].groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()   

z_fd = df[['perc_full_branch']].groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
z_ls_fd = df_ls[['perc_full_branch']].groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()  
#----------------------------------------------
#----------------------------------------------    
# MODEL 1: FD IV, charge-off rates
#----------------------------------------------
#----------------------------------------------

#----------------------------------------------
# STEP 1: OLS, with clustered standard errors
#----------------------------------------------

# Estimate G_hat
mod1_step1 = PanelOLS(w_fd, pd.concat([x_fd,z_fd],axis = 1))
res1_step1 = mod1_step1.fit(cov_type = 'clustered', cluster_entity = True)
print(res1_step1.summary)
G_hat_fd = res1_step1.fitted_values

# Calculate G_hat_x_xbar
G_hat_x_xbar_fd = x_xbar_fd * G_hat_fd.values

#----------------------------------------------
# STEP 2: OLS, with clustered standard errors
#----------------------------------------------

# setup the indepedent variables
x_mod1a = pd.concat([x_fd,G_hat_fd],axis = 1)
x_mod1b = pd.concat([x_fd,G_hat_fd,G_hat_x_xbar_fd],axis = 1)

mod1a_step2 = PanelOLS(y_charge_fd,x_mod1a)
res1a_step2 = mod1a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res1a_step2.summary)

mod1b_step2 = PanelOLS(y_charge_fd,x_mod1b)
res1b_step2 = mod1b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res1b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test weak instruments
## F-test
### First calculate the reduced model (without instruments)
res1_step1b = PanelOLS(w_fd, pd.concat([x_fd],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)

def fTestWeakInstruments(y, fitted_full, fitted_reduced, dof = 4):
    ''' Simple F-test to test the strength of instrumental variables'''
    
    # Calculate the SSE and MSE
    sse_full = np.sum([(y.values[i] - fitted_full.values[i][0])**2 for i in range(y.shape[0])])
    sse_reduced =  np.sum([(y.values[i] - fitted_reduced.values[i][0])**2 for i in range(y.shape[0])])
    
    mse_full = (1 / y.shape[0]) * np.sum([(y.values[i] - fitted_full.values[i][0])**2 for i in range(y.shape[0])])
    
    # Calculate the statistic
    f_stat = ((sse_reduced - sse_full)/dof) / mse_full
    
    return f_stat

f_test_mod1 = fTestWeakInstruments(w_fd, res1_step1.fitted_values, res1_step1b.fitted_values, 2) #10.2021

# Test for endogeneity
mod1a_endo = PanelOLS(y_charge_fd,pd.concat([x_fd,w_fd,res1_step1.resids],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod1a_endo.summary) #p-value = 0.2131

# Test for overidentifying restrictions
oir_mod1a = sargan(res1a_step2.resids, x_fd, z_fd)
oir_mod1b = sargan(res1b_step2.resids, x_fd, z_fd)
'''NOTE: Both tests are insignificant'''

#----------------------------------------------
#----------------------------------------------    
# MODEL 2: FD IV, allowance rates
#----------------------------------------------
#----------------------------------------------

#----------------------------------------------
# STEP 2: OLS, with clustered standard errors
#----------------------------------------------

# setup the indepedent variables
mod2a_step2 = PanelOLS(y_allow_fd,x_mod1a)
res2a_step2 = mod2a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res2a_step2.summary)

mod2b_step2 = PanelOLS(y_allow_fd,x_mod1b)
res2b_step2 = mod2b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res2b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test for endogeneity
mod2a_endo = PanelOLS(y_allow_fd,pd.concat([x_fd,w_fd,res1_step1.resids],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod2a_endo.summary) #p-value = 0.4842

# Test for overidentifying restrictions
oir_mod2a = sargan(res2a_step2.resids, x_fd, z_fd)
oir_mod2b = sargan(res2b_step2.resids, x_fd, z_fd)
'''NOTE: Both tests are significant. Instruments not good.'''
    
#----------------------------------------------
#----------------------------------------------    
# MODEL 3: FD IV, charge-off rates, subsample
#----------------------------------------------
#----------------------------------------------
        
#----------------------------------------------
# STEP 1: OLS, with clustered standard errors
#----------------------------------------------

# Estimate G_hat
mod3_step1 = PanelOLS(w_ls_fd, pd.concat([x_ls_fd,z_ls_fd],axis = 1))
res3_step1 = mod3_step1.fit(cov_type = 'clustered', cluster_entity = True)
print(res3_step1.summary)
G_hat_fd = res3_step1.fitted_values

# Calculate G_hat_x_xbar
G_hat_x_xbar_fd = x_xbar_ls_fd * G_hat_fd.values

#----------------------------------------------
# STEP 2: OLS, with clustered standard errors
#----------------------------------------------

# setup the indepedent variables
x_mod3a = pd.concat([x_ls_fd,G_hat_fd],axis = 1)
x_mod3b = pd.concat([x_ls_fd,G_hat_fd,G_hat_x_xbar_fd],axis = 1)

mod3a_step2 = PanelOLS(y_charge_ls_fd,x_mod3a)
res3a_step2 = mod3a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res3a_step2.summary)

mod3b_step2 = PanelOLS(y_charge_ls_fd,x_mod3b)
res3b_step2 = mod3b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res3b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test weak instruments
## F-test
### First calculate the reduced model (without instruments)
res3_step1b = PanelOLS(w_ls_fd, x_ls_fd).fit(cov_type = 'clustered', cluster_entity = True)

f_test_mod3 = fTestWeakInstruments(w_ls_fd, res3_step1.fitted_values, res3_step1b.fitted_values, 2) #4.4882

# Test for endogeneity
mod3a_endo = PanelOLS(y_charge_ls_fd,pd.concat([x_ls_fd,w_ls_fd,res3_step1.resids],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod3a_endo.summary) #p-value = 0.4393

# Test for overidentifying restrictions
oir_mod3a = sargan(res3a_step2.resids, x_ls_fd, z_ls_fd)
oir_mod3b = sargan(res3b_step2.resids, x_ls_fd, z_ls_fd)
'''NOTE: Both tests are insignificant'''

#----------------------------------------------
#----------------------------------------------    
# MODEL 4: FD IV, Allow rates, subsample
#----------------------------------------------
#----------------------------------------------

#----------------------------------------------
# STEP 2: OLS, with clustered standard errors
#----------------------------------------------

mod4a_step2 = PanelOLS(y_allow_ls_fd,x_mod3a)
res4a_step2 = mod4a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res4a_step2.summary)

mod4b_step2 = PanelOLS(y_allow_ls_fd,x_mod3b)
res4b_step2 = mod4b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res4b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test for endogeneity
mod4a_endo = PanelOLS(y_allow_ls_fd,pd.concat([x_ls_fd,w_ls_fd,res3_step1.resids],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod4a_endo.summary) #p-value = 0.2226

# Test for overidentifying restrictions
oir_mod4a = sargan(res4a_step2.resids, x_ls_fd, z_ls_fd)
oir_mod4b = sargan(res4b_step2.resids, x_ls_fd, z_ls_fd)
'''NOTE: Both tests are insignificant'''

#----------------------------------------------
#----------------------------------------------    
# Make a nice table for model 1 and 2
#----------------------------------------------
#----------------------------------------------
# Make tables
table_fd_step1 = summary_col([res1_step1, res3_step1], show = 'se', regressor_order = ['distance','num_branch','perc_full_branch','STALPBR'])
table_fd_step2 = summary_col([res1a_step2, res3a_step2,res1b_step2, res3b_step2,\
                              res2a_step2, res4a_step2,res2b_step2, res4b_step2],\
                     show = 'se', regressor_order = ['fitted_values','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio'])

# Save to a single excel
table_fd_step1.to_excel('FD_IV_fd_lsoverta_step1.xlsx')
table_fd_step2.to_excel('FD_IV_fd_lsoverta_step2.xlsx')