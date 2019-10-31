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

#----------------------------------------------
# Prelims

## Select and setup the variables
### Dependent variable step 1
w = df.dum_ls

### Dependent variables step 2
y_charge = df.net_coffratio_tot_ta
y_allow = df.allowratio_tot_ta

### Independent exogenous variables
x = df[['cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio']]
x = add_constant(x)

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
# MODEL 1: POOLED IV, charge-off rates, no dynamic effects
#----------------------------------------------
#----------------------------------------------
    
#----------------------------------------------
# STEP 1: Pooled Probit
#----------------------------------------------

# Estimate G_hat
mod1_step1 = Probit(w, pd.concat([x,z],axis = 1))
res1_step1 = mod1_step1.fit(cov_type = 'HC1')
print(res1_step1.summary())
G_hat = res1_step1.fittedvalues.rename('G_hat')

# Calculate G_hat_x_xbar
G_hat_x_xbar = x_xbar.T.multiply(G_hat).T

#----------------------------------------------
# STEP 2: Pooled OLS, with clustered standard errors
#----------------------------------------------

# setup the indepedent variables
x_mod1a = pd.concat([x,G_hat],axis = 1)
x_mod1b = pd.concat([x,G_hat,G_hat_x_xbar],axis = 1)

mod1a_step2 = PanelOLS(y_charge,x_mod1a)
res1a_step2 = mod1a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res1a_step2.summary)

mod1b_step2 = PanelOLS(y_charge,x_mod1b)
res1b_step2 = mod1b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res1b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test weak instruments
## Prelims
hypotheses = '(num_branch = 0),(perc_full_branch = 0),(STALPBR = 0),(distance = 0)'

## F-test
weak_instr_f = res1_step1.f_test(hypotheses) #p ~= 0.00

## Wald test
weak_instr_wald = res1_step1.wald_test(hypotheses) #p ~= 0.00

# Test for endogeneity
mod1a_endo = PanelOLS(y_charge,pd.concat([x,w,res1_step1.resid_generalized],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod1a_endo.summary) #p-value = 0.0001

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

#----------------------------------------------
#----------------------------------------------    
# MODEL 2: POOLED IV, allow-off rates, no dynamic effects
#----------------------------------------------
#----------------------------------------------
    
#----------------------------------------------
# STEP 1: Pooled Probit
#----------------------------------------------

# Identical to model 1

#----------------------------------------------
# STEP 2: Pooled OLS, with clustered standard errors
#----------------------------------------------

# setup the indepedent variables
x_mod2a = pd.concat([x,G_hat],axis = 1)
x_mod2b = pd.concat([x,G_hat,G_hat_x_xbar],axis = 1)

mod2a_step2 = PanelOLS(y_allow,x_mod2a)
res2a_step2 = mod2a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res2a_step2.summary)

mod2b_step2 = PanelOLS(y_allow,x_mod2b)
res2b_step2 = mod2b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res2b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test weak instruments
## Identical to model 1

# Test for endogeneity
mod2_endo = PanelOLS(y_allow,pd.concat([x,w,res1_step1.resid_generalized],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod2_endo.summary) #p-value = 0.7731

# Test for overidentifying restrictions
oir_mod2a = sargan(res2a_step2.resids, x, z)
oir_mod2b = sargan(res2b_step2.resids, x, z)

#----------------------------------------------
#----------------------------------------------    
# Make a nice table for model 1 and 2
#----------------------------------------------
#----------------------------------------------
# Make tables
table_step1 = summary_col([res1_step1], show = 'se', regressor_order = ['distance','num_branch','perc_full_branch','STALPBR'])
table_step2 = summary_col([res1a_step2,res1b_step2,res2a_step2,res2b_step2],\
                     show = 'se', regressor_order = ['G_hat','cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio'])

# Save to a single excel
table_step1.to_excel('Pooled_IV_step1.xlsx')
table_step2.to_excel('Pooled_IV_step2.xlsx')

#----------------------------------------------
#----------------------------------------------    
# MODEL 3: POOLED Control Function approach, charge-off rates, no dynamic effects
#----------------------------------------------
#----------------------------------------------

#----------------------------------------------
# STEP 1: Pooled Probit
#----------------------------------------------

# Probit is the same as model 1 and 2

# Post estimation
## Calculate capital and lowercase phi
Phi_hat = sps.norm.cdf(res1_step1.fittedvalues)
phi_hat = sps.norm.pdf(res1_step1.fittedvalues)

## Inverse mills ratios
inv_mills = phi_hat / Phi_hat
inv_mills_alt = phi_hat / (1 - Phi_hat)

## Variables to be used in estimation
Phi_hat_x_xbar =  x_xbar.T.multiply(Phi_hat).T
w_x_xbar = x_xbar.T.multiply(w).T
w_inv_mills = w * inv_mills
w_inv_mills.rename('w_inv_mills', inplace = True)
w_inv_mills_alt = (1 - w) * inv_mills_alt
w_inv_mills_alt.rename('w_inv_mills_alt', inplace = True)

#----------------------------------------------
# STEP 2: Pooled OLS, with clustered standard errors
#----------------------------------------------

# setup the indepedent variables
x_mod3a = pd.concat([x,Phi_hat_x_xbar,pd.Series(Phi_hat, name = 'Phi_hat', index = x.index),\
                     pd.Series(phi_hat, name = 'phi_hat', index = x.index)],axis = 1)
x_mod3b = pd.concat([x,w,w_x_xbar,w_inv_mills,w_inv_mills_alt],axis = 1)

mod3a_step2 = PanelOLS(y_charge,x_mod3a)
res3a_step2 = mod3a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res3a_step2.summary)

mod3b_step2 = PanelOLS(y_charge,x_mod3b)
res3b_step2 = mod3b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res3b_step2.summary)

#----------------------------------------------
# Test Assumptions model
#----------------------------------------------

# Wald test on the last two variables  model 3b
wald_mod3 = res3b_step2.wald_test(formula = 'w_inv_mills = w_inv_mills_alt = 0')

#----------------------------------------------
#----------------------------------------------    
# Make a nice table for model 3
#----------------------------------------------
#----------------------------------------------
# Make tables
table_mod3 = summary_col([res3a_step2,res3b_step2],\
                     show = 'se', regressor_order = ['Phi_hat','dum_ls'])

# Save to a single excel
table_mod3.to_excel('Correction_control_function.xlsx')

#----------------------------------------------
#----------------------------------------------    
# MODEL 4: FD IV, charge-off rates
#----------------------------------------------
#----------------------------------------------

# Prelims
## Transform variables
w_fd = w.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
w_fd_alt = df.ls_tot.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
x_fd = add_constant(df[['cd_pur_ta','cd_sold_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio']].groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna())
z_fd = z.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
x_xbar_fd = x_xbar.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

y_charge_fd = y_charge.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
    
#----------------------------------------------
# STEP 1: OLS, with clustered standard errors
#----------------------------------------------

# Estimate G_hat
mod4_step1 = PanelOLS(w_fd, pd.concat([x_fd,z_fd],axis = 1))
res4_step1 = mod4_step1.fit(cov_type = 'clustered', cluster_entity = True)
print(res4_step1.summary)
G_hat_fd = res4_step1.fitted_values

# Calculate G_hat_x_xbar
G_hat_x_xbar_fd = x_xbar_fd * G_hat_fd.values

#----------------------------------------------
# STEP 2: OLS, with clustered standard errors
#----------------------------------------------

# setup the indepedent variables
x_mod4a = pd.concat([x_fd,G_hat_fd],axis = 1)
x_mod4b = pd.concat([x_fd,G_hat_fd,G_hat_x_xbar_fd],axis = 1)

mod4a_step2 = PanelOLS(y_charge_fd,x_mod4a)
res4a_step2 = mod4a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res4a_step2.summary)

mod4b_step2 = PanelOLS(y_charge_fd,x_mod4b)
res4b_step2 = mod4b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res4b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test for endogeneity
mod1a_endo = PanelOLS(y_charge,pd.concat([x,w,res1_step1.resid_generalized],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod1a_endo.summary) #p-value = 0.0001

# Test for overidentifying restrictions
oir_mod4a = sargan(res4a_step2.resids, x_fd, z_fd)
oir_mod4b = sargan(res4b_step2.resids, x_fd, z_fd)

