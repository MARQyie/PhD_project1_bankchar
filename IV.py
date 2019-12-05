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

# Load df
df = pd.read_csv('df_wp1_clean.csv', index_col = 0)

# Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip() + '1230')
df.set_index(['IDRSSD','date'],inplace=True)

# Drop missings on distance
df.dropna(subset = ['distance'], inplace = True)

# Dummy variable for loan sales
df['dum_ls'] = np.exp((df.ls_tot > 0) * 1) - 1 #will be taken the log of later
#dum_ls = (df.ls_tot > 0) * 1

# Make variables that determines the group
#df['group'] = (df.index.get_level_values(0).isin(df[dum_ls == 1].index.get_level_values(0).unique().to_list())) * 1
#df['group'] = (df.index.get_level_values(0).isin(df[np.log(1 + df['dum_ls']) == 1].index.get_level_values(0).unique().to_list())) * 1

# Subset the df
#df_ls = df[df.group == 1]

## Kick out the community banks (based on Stiroh, 2004)   
df_ls = df[~((df.RC2170 < 3e5) & (df.bhc == 0))]

## Only take the banks that change in dum_ls 
#intersect = np.intersect1d(df[df.ls_tot > 0].index.get_level_values(0).unique(), df[df.ls_tot == 0].index.get_level_values(0).unique())
#df_ls = df[df.index.get_level_values(0).isin(intersect)]
#dum_ls_ls = (df_ls.ls_tot > 0) * 1

## Take logs
df = df.select_dtypes(include = ['float64','int']).transform(lambda df: np.log(1 + df)).replace([np.inf, -np.inf], 0)
df_ls = df_ls.select_dtypes(include = ['float64','int']).transform(lambda df: np.log(1 + df)).replace([np.inf, -np.inf], 0)

## Drop NaNs on subset
df.dropna(subset = ['provratio','rwata','net_coffratio_tot_ta','allowratio_tot_ta','ls_tot_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170'], inplace = True)
df_ls.dropna(subset = ['provratio','rwata','net_coffratio_tot_ta','allowratio_tot_ta','ls_tot_ta',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170'], inplace = True)

## Add the dummy variable to the dfs
#df['dum_ls'] = dum_ls
#df_ls['dum_ls'] = dum_ls_ls

#----------------------------------------------
# Prelims

## Select and setup the variables
### Dependent variable step 1
w = df.dum_ls

### Dependent variables step 2
y_charge = df.net_coffratio_tot_ta
y_allow = df.allowratio_tot_ta
y_rwata = df.rwata
y_prov = df.provratio

### Independent exogenous variables
x = df[[\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']]
x = add_constant(x)

x_xbar = df[['RC7205','loanratio','roa','depratio','comloanratio','RC2170']].transform(lambda df: df - df.mean())

#### Change the columns of x_xbar
dict_x_xbar = dict(zip(['RC7205','loanratio','roa','depratio','comloanratio','RC2170'],\
         [x + '_xbar' for x in ['RC7205','loanratio','roa','depratio','comloanratio','RC2170']]))
x_xbar.rename(columns = dict_x_xbar, inplace = True)

### Instruments
#z = df[['num_branch', 'perc_full_branch', 'STALPBR', 'distance']]
#z_alt = df[['num_branch', 'perc_limited_branch', 'STALPBR', 'distance']]

z = df[['perc_full_branch']]


# Functions

def sargan(resids, x, z, nendog = 1):

    nobs, ninstr = z.shape
    name = 'Sargan\'s test of overidentification'

    eps = resids.values[:,None]
    u = annihilate(eps, pd.concat([x,z],axis = 1))
    stat = nobs * (1 - (u.T @ u) / (eps.T @ eps)).squeeze()
    null = 'The overidentification restrictions are valid'

    return WaldTestStatistic(stat, null, ninstr - nendog, name=name)
  
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
#hypotheses = '(num_branch = 0),(perc_full_branch = 0),(STALPBR = 0),(distance = 0)'

## F-test
#weak_instr_f = res1_step1.f_test(hypotheses) #p ~= 0.00

## Wald test
#weak_instr_wald = res1_step1.wald_test(hypotheses) #p ~= 0.00

# Test for endogeneity
mod1a_endo = PanelOLS(y_charge,pd.concat([x,w,res1_step1.resid_generalized],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod1a_endo.summary) #p-value = 0.0001

# Test for overidentifying restrictions


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
                     show = 'se', regressor_order = ['G_hat',\
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
x_fd = df[[\
               'RC7205','loanratio','roa','depratio','comloanratio','size']].groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
z_fd = z.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
x_xbar_fd = x_xbar.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

y_charge_fd = y_charge.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
y_allow_fd = y_allow.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
y_rwata_fd = y_rwata.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
y_prov_fd = y_prov.groupby(df.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
        
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

# Test weak instruments
## F-test
### First calculate the reduced model (without instruments)
res4_step1b = PanelOLS(w_fd, x_fd).fit(cov_type = 'clustered', cluster_entity = True)

def fTestWeakInstruments(y, fitted_full, fitted_reduced, dof = 4):
    ''' Simple F-test to test the strength of instrumental variables'''
    
    # Calculate the SSE and MSE
    sse_full = np.sum([(y.values[i] - fitted_full.values[i][0])**2 for i in range(y.shape[0])])
    sse_reduced =  np.sum([(y.values[i] - fitted_reduced.values[i][0])**2 for i in range(y.shape[0])])
    
    mse_full = (1 / y.shape[0]) * np.sum([(y.values[i] - fitted_full.values[i][0])**2 for i in range(y.shape[0])])
    
    # Calculate the statistic
    f_stat = ((sse_reduced - sse_full)/dof) / mse_full
    
    return f_stat

f_test_mod4 = fTestWeakInstruments(w_fd, res4_step1.fitted_values, res4_step1b.fitted_values, 2) #11.284976283499443

# Test for endogeneity
mod4a_endo = PanelOLS(y_charge_fd,pd.concat([x_fd,w_fd,res4_step1.resids],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod4a_endo.summary) #p-value = 0.01

# Test for overidentifying restrictions
oir_mod4a = sargan(res4a_step2.resids, x_fd, z_fd)
oir_mod4b = sargan(res4b_step2.resids, x_fd, z_fd)
'''NOTE: Both tests are significant'''

#----------------------------------------------
#----------------------------------------------    
# MODEL 5: FD IV, allowance rates
#----------------------------------------------
#----------------------------------------------

#----------------------------------------------
# STEP 2: OLS, with clustered standard errors
#----------------------------------------------

# setup the indepedent variables
mod5a_step2 = PanelOLS(y_allow_fd,x_mod4a)
res5a_step2 = mod5a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res5a_step2.summary)

mod5b_step2 = PanelOLS(y_allow_fd,x_mod4b)
res5b_step2 = mod5b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res5b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test for endogeneity
mod5a_endo = PanelOLS(y_allow_fd,pd.concat([x_fd,w_fd,res4_step1.resids],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod5a_endo.summary) #p-value = 0.0000

# Test for overidentifying restrictions
oir_mod5a = sargan(res5a_step2.resids, x_fd, z_fd)
oir_mod5b = sargan(res5b_step2.resids, x_fd, z_fd)
'''NOTE: Both tests are significant at 5%'''
    
#----------------------------------------------
#----------------------------------------------    
# MODEL 6: FD IV, charge-off rates, subsample
#----------------------------------------------
#----------------------------------------------

# Prelims
## Set variables
### Dummy LS
w_ls_fd = df_ls.dum_ls.groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

#### Dependent variables step 2
y_charge_ls_fd = df_ls.net_coffratio_tot_ta.groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
y_allow_ls_fd = df_ls.allowratio_tot_ta.groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
y_rwata_ls_fd = df_ls.rwata.groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
y_prov_ls_fd = df_ls.provratio.groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

### Independent exogenous variables
x_ls_fd = df_ls[[\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170']].groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

x_xbar_ls_fd = df_ls[['RC7205','loanratio','roa','depratio','comloanratio','RC2170']].transform(lambda df: df - df.mean()).groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

#### Change the columns of x_xbar
dict_x_xbar = dict(zip(['RC7205','loanratio','roa','depratio','comloanratio','RC2170'],\
         [x + '_xbar' for x in ['cd_pur_ta','cd_sold_ta','RC7205','loanratio','roa','depratio','comloanratio','RC2170']]))
x_xbar_ls_fd.rename(columns = dict_x_xbar, inplace = True)

### Instruments
#z_ls_fd = df_ls[['num_branch', 'perc_full_branch', 'STALPBR', 'distance']].groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
z_ls_fd = df_ls[['perc_full_branch']].groupby(df_ls.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()
        
#----------------------------------------------
# STEP 1: OLS, with clustered standard errors
#----------------------------------------------

# Estimate G_hat
mod6_step1 = PanelOLS(w_ls_fd, pd.concat([x_ls_fd,z_ls_fd],axis = 1))
res6_step1 = mod6_step1.fit(cov_type = 'clustered', cluster_entity = True)
print(res6_step1.summary)
G_hat_fd = res6_step1.fitted_values

# Calculate G_hat_x_xbar
G_hat_x_xbar_fd = x_xbar_ls_fd * G_hat_fd.values

#----------------------------------------------
# STEP 2: OLS, with clustered standard errors
#----------------------------------------------

# setup the indepedent variables
x_mod6a = pd.concat([x_ls_fd,G_hat_fd],axis = 1)
x_mod6b = pd.concat([x_ls_fd,G_hat_fd,G_hat_x_xbar_fd],axis = 1)

mod6a_step2 = PanelOLS(y_charge_ls_fd,x_mod6a)
res6a_step2 = mod6a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res6a_step2.summary)

mod6b_step2 = PanelOLS(y_charge_ls_fd,x_mod6b)
res6b_step2 = mod6b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res6b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test weak instruments
## F-test
### First calculate the reduced model (without instruments)
res6_step1b = PanelOLS(w_ls_fd, x_ls_fd).fit(cov_type = 'clustered', cluster_entity = True)

f_test_mod6 = fTestWeakInstruments(w_ls_fd, res6_step1.fitted_values, res6_step1b.fitted_values, 2) #2.8017

# Test for endogeneity
mod6a_endo = PanelOLS(y_charge_ls_fd,pd.concat([x_ls_fd,w_ls_fd,res6_step1.resids],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod6a_endo.summary) #p-value = 0.0212

# Test for overidentifying restrictions
oir_mod6a = sargan(res6a_step2.resids, x_ls_fd, z_ls_fd)
oir_mod6b = sargan(res6b_step2.resids, x_ls_fd, z_ls_fd)
'''NOTE: Both tests are insignificant'''

#----------------------------------------------
#----------------------------------------------    
# MODEL 7: FD IV, Allow rates
#----------------------------------------------
#----------------------------------------------

#----------------------------------------------
# STEP 2: OLS, with clustered standard errors
#----------------------------------------------

mod7a_step2 = PanelOLS(y_allow_ls_fd,x_mod6a)
res7a_step2 = mod7a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res7a_step2.summary)

mod7b_step2 = PanelOLS(y_allow_ls_fd,x_mod6b)
res7b_step2 = mod7b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res7b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test for endogeneity
mod7a_endo = PanelOLS(y_allow_ls_fd,pd.concat([x_ls_fd,w_ls_fd,res6_step1.resids],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod7a_endo.summary) #p-value = 0.0000

# Test for overidentifying restrictions
oir_mod7a = sargan(res7a_step2.resids, x_ls_fd, z_ls_fd)
oir_mod7b = sargan(res7b_step2.resids, x_ls_fd, z_ls_fd)
'''NOTE: Both tests are insignificant'''

#----------------------------------------------
#----------------------------------------------    
# MODEL 8: FD IV, rwata
#----------------------------------------------
#----------------------------------------------

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
x_mod8a = pd.concat([x_fd,G_hat_fd],axis = 1)
x_mod8b = pd.concat([x_fd,G_hat_fd,G_hat_x_xbar_fd],axis = 1)

mod8a_step2 = PanelOLS(y_rwata_fd,x_mod8a)
res8a_step2 = mod8a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res8a_step2.summary)

mod8b_step2 = PanelOLS(y_rwata_fd,x_mod8b)
res8b_step2 = mod8b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res8b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test for endogeneity
mod8a_endo = PanelOLS(y_rwata_fd,pd.concat([x_fd,w_fd,res4_step1.resids],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod8a_endo.summary) #p-value = 0.0000

# Test for overidentifying restrictions
oir_mod8a = sargan(res8a_step2.resids, x_fd, z_fd)
oir_mod8b = sargan(res8b_step2.resids, x_fd, z_fd)
'''NOTE: Both tests are insignificant'''

#----------------------------------------------
#----------------------------------------------    
# MODEL 9: FD IV, prov
#----------------------------------------------
#----------------------------------------------

#----------------------------------------------
# STEP 1: OLS, with clustered standard errors
#----------------------------------------------

# Same as model 4

#----------------------------------------------
# STEP 2: OLS, with clustered standard errors
#----------------------------------------------

# setup the indepedent variables
x_mod9a = pd.concat([x_fd,G_hat_fd],axis = 1)
x_mod9b = pd.concat([x_fd,G_hat_fd,G_hat_x_xbar_fd],axis = 1)

mod9a_step2 = PanelOLS(y_prov_fd,x_mod9a)
res9a_step2 = mod9a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res9a_step2.summary)

mod9b_step2 = PanelOLS(y_prov_fd,x_mod9b)
res9b_step2 = mod9b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res8b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test for endogeneity
mod9a_endo = PanelOLS(y_prov_fd,pd.concat([x_fd,w_fd,res4_step1.resids],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod9a_endo.summary) #p-value = 0.1663

# Test for overidentifying restrictions
oir_mod9a = sargan(res9a_step2.resids, x_fd, z_fd)
oir_mod9b = sargan(res9b_step2.resids, x_fd, z_fd)
'''NOTE: Both tests are insignificant'''

#----------------------------------------------
#----------------------------------------------    
# MODEL 10: FD IV, rwata, subsample
#----------------------------------------------
#----------------------------------------------
        
#----------------------------------------------
# STEP 1: OLS, with clustered standard errors
#----------------------------------------------

# Estimate G_hat
mod6_step1 = PanelOLS(w_ls_fd, pd.concat([x_ls_fd,z_ls_fd],axis = 1))
res6_step1 = mod6_step1.fit(cov_type = 'clustered', cluster_entity = True)
print(res6_step1.summary)
G_hat_fd = res6_step1.fitted_values

# Calculate G_hat_x_xbar
G_hat_x_xbar_fd = x_xbar_ls_fd * G_hat_fd.values

#----------------------------------------------
# STEP 2: OLS, with clustered standard errors
#----------------------------------------------

# setup the indepedent variables
x_mod10a = pd.concat([x_ls_fd,G_hat_fd],axis = 1)
x_mod10b = pd.concat([x_ls_fd,G_hat_fd,G_hat_x_xbar_fd],axis = 1)

mod10a_step2 = PanelOLS(y_rwata_ls_fd,x_mod10a)
res10a_step2 = mod10a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res10a_step2.summary)

mod10b_step2 = PanelOLS(y_rwata_ls_fd,x_mod10b)
res10b_step2 = mod10b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res10b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test for endogeneity
mod10a_endo = PanelOLS(y_rwata_ls_fd,pd.concat([x_ls_fd,w_ls_fd,res6_step1.resids],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod10a_endo.summary) #p-value = 0.0002

# Test for overidentifying restrictions
oir_mod10a = sargan(res10a_step2.resids, x_ls_fd, z_ls_fd)
oir_mod10b = sargan(res10b_step2.resids, x_ls_fd, z_ls_fd)
'''NOTE: Both tests are insignificant'''

#----------------------------------------------
#----------------------------------------------    
# MODEL 11: FD IV, prov, subsample
#----------------------------------------------
#----------------------------------------------
        
#----------------------------------------------
# STEP 1: OLS, with clustered standard errors
#----------------------------------------------

# Same as model 6

#----------------------------------------------
# STEP 2: OLS, with clustered standard errors
#----------------------------------------------

# setup the indepedent variables
x_mod11a = pd.concat([x_ls_fd,G_hat_fd],axis = 1)
x_mod11b = pd.concat([x_ls_fd,G_hat_fd,G_hat_x_xbar_fd],axis = 1)

mod11a_step2 = PanelOLS(y_prov_ls_fd,x_mod11a)
res11a_step2 = mod11a_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res11a_step2.summary)

mod11b_step2 = PanelOLS(y_prov_ls_fd,x_mod11b)
res11b_step2 = mod11b_step2.fit(cov_type = 'clustered', cluster_entity = True)
print(res11b_step2.summary)

#----------------------------------------------
# Test instruments
#----------------------------------------------
# Test for endogeneity
mod11a_endo = PanelOLS(y_prov_ls_fd,pd.concat([x_ls_fd,w_ls_fd,res6_step1.resids],axis = 1)).fit(cov_type = 'clustered', cluster_entity = True)
print(mod11a_endo.summary) #p-value = 0.8449

# Test for overidentifying restrictions
oir_mod11a = sargan(res11a_step2.resids, x_ls_fd, z_ls_fd)
oir_mod11b = sargan(res11b_step2.resids, x_ls_fd, z_ls_fd)
'''NOTE: Both tests are insignificant'''

#----------------------------------------------
#----------------------------------------------    
# Make a nice table for model 1 and 2
#----------------------------------------------
#----------------------------------------------
# Make tables
table_fd_step1 = summary_col([res4_step1, res6_step1], show = 'se', regressor_order = ['distance','num_branch','perc_full_branch','STALPBR'])
table_fd_step2 = summary_col([res4a_step2, res6a_step2,res4b_step2, res6b_step2,\
                              res5a_step2, res7a_step2,res5b_step2, res7b_step2,\
                              res8a_step2, res10a_step2,res8b_step2, res10b_step2,\
                              res9a_step2, res11a_step2,res9b_step2, res11b_step2],\
                     show = 'se', regressor_order = ['fitted_values',\
               'RC7205','loanratio','roa','depratio','comloanratio'])

# Save to a single excel
table_fd_step1.to_excel('FD_IV_restricted_sample_step1.xlsx')
table_fd_step2.to_excel('FD_IV_restricted_sample_step2.xlsx')