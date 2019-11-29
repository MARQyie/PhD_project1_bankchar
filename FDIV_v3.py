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

import itertools
#--------------------------------------------
# Set parameters 
log = False # If set to False the program estimates the model without logs and with size

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
#----------------------------------------------
# Prelims
#----------------------------------------------
#----------------------------------------------

#----------------------------------------------
# Load data and add needed variables

# Load df
df = pd.read_csv('df_wp1_clean.csv', index_col = 0)

## Make multi index
df.date = pd.to_datetime(df.date.astype(str).str.strip() + '1230')
df.set_index(['IDRSSD','date'],inplace=True)

## Drop missings on distance
df.dropna(subset = ['distance'], inplace = True)

## Dummy variable for loan sales
if log:
    df['dum_ls'] = np.exp((df.ls_tot > 0) * 1) - 1 #will be taken the log of later
else:
    df['dum_ls'] = (df.ls_tot > 0) * 1    

## Take a subset of variables (only the ones needed)
vars_needed = ['distance','provratio','rwata','net_coffratio_tot_ta',\
               'allowratio_tot_ta','ls_tot_ta','dum_ls','size',\
               'RC7205','loanratio','roa','depratio','comloanratio','RC2170',\
               'num_branch', 'bhc', 'RIAD4150', 'perc_limited_branch',\
               'perc_full_branch', 'unique_states','UNIT']
df_full = df[vars_needed]

## drop NaNs
df_full.dropna(subset = ['provratio','rwata','net_coffratio_tot_ta','allowratio_tot_ta',\
               'ls_tot_ta','RC7205','loanratio','roa',\
               'depratio','comloanratio','RC2170','size'], inplace = True)

#---------------------------------------------------
# Setup the data

## Set aside TA
ta = df_full.RC2170

## Correct dummy and percentage variables for log
if log:
    df_full['bhc'] = np.exp(df_full.bhc) - 1
    df_full['UNIT'] = np.exp(df_full.UNIT) - 1

## Take logs of the df
if log:
    df_full = df_full.transform(lambda df: np.log(1 + df)).replace([np.inf, -np.inf], 0)

## Add TA for subsetting
df_full['ta_sub'] = ta

## Add the x_xbar to the df
if log:
    x_xbar = df_full[['RC7205','loanratio','roa',\
                      'depratio','comloanratio','RC2170','bhc']].transform(lambda df: df - df.mean())
    df_full[[x + '_xbar' for x in ['RC7205','loanratio',\
                                   'roa','depratio','comloanratio','RC2170','bhc']]] = x_xbar
else:
    x_xbar = df_full[['RC7205','loanratio','roa',\
                          'depratio','comloanratio','size','bhc']].transform(lambda df: df - df.mean())
    df_full[[x + '_xbar' for x in ['RC7205','loanratio',\
                                   'roa','depratio','comloanratio','size','bhc']]] = x_xbar

## Take the first differences
df_full_fd = df_full.groupby(df_full.index.get_level_values(0)).transform(lambda df: df.shift(periods = 1) - df).dropna()

## Add dummies
dummy_full_fd = pd.get_dummies(df_full_fd.index.get_level_values(1))

### Add dummies to the dfs
col_dummy = ['dum' + dummy for dummy in dummy_full_fd.columns.astype(str).str[:4].tolist()]
dummy_full_fd = pd.DataFrame(np.array(dummy_full_fd), index = df_full_fd.index, columns = col_dummy)
df_full_fd[col_dummy] = dummy_full_fd

# Subset the df take the crisis subsets
''' Crisis dates are:
    Pre-crisis: 2001-2006
    Crisis: 2007-2009
    Post-crisis: 2010-2018
    '''

df_pre_fd = df_full_fd[df_full_fd.index.get_level_values(1) <= pd.Timestamp(2006,12,30)]
df_during_fd = df_full_fd[(df_full_fd.index.get_level_values(1) > pd.Timestamp(2006,12,30)) & (df_full_fd.index.get_level_values(1) < pd.Timestamp(2010,12,30))]
df_post_fd = df_full_fd[df_full_fd.index.get_level_values(1) >= pd.Timestamp(2010,12,30)]

#---------------------------------------------------
# Load the necessary functions

def fTestWeakInstruments(y, fitted_full, fitted_reduced, dof = 4):
    ''' Simple F-test to test the strength of instrumental variables.
        
        y : True y values
        fitted_full : fitted values of the first stage with instruments
        fitted_reduced : fitted values first stage without instruments
        dof : number of instruments'''
    
    # Calculate the SSE and MSE
    sse_full = np.sum([(y.values[i] - fitted_full.values[i][0])**2 for i in range(y.shape[0])])
    sse_reduced =  np.sum([(y.values[i] - fitted_reduced.values[i][0])**2 for i in range(y.shape[0])])
    
    mse_full = (1 / y.shape[0]) * np.sum([(y.values[i] - fitted_full.values[i][0])**2 for i in range(y.shape[0])])
    
    # Calculate the statistic
    f_stat = ((sse_reduced - sse_full)/dof) / mse_full
    
    return f_stat

def sargan(resids, x, z, nendog = 1):
    '''Function performs a sargan test (no heteroskedasity) to check
        the validity of the overidentification restrictions. H0: 
        overidentifying restrictions hold.
        
        resids : residuals of the second stage
        x : exogenous variables
        z : instruments 
        nendog : number of endogenous variables'''  

    nobs, ninstr = z.shape
    name = 'Sargan\'s test of overidentification'

    eps = resids.values[:,None]
    u = annihilate(eps, pd.concat([x,z],axis = 1))
    stat = nobs * (1 - (u.T @ u) / (eps.T @ eps)).squeeze()
    null = 'The overidentification restrictions are valid'

    return WaldTestStatistic(stat, null, ninstr - nendog, name=name)
            
'''-----------------------------------------''' 
#----------------------------------------------
# Setup the method that does the FDIV plus tests
#----------------------------------------------
'''-----------------------------------------''' 

def analysesFDIV(df, var_ls, righthand_x, righthand_ghat, righthand_z, time_dummies):
    ''' Performs a FDIV linear model. The second stage takes for dependent variables:
        1) Charge-off rates, 2) Loan Loss allowances, 3) RWA/TA 4) Loan loss provisions.
        The method also correct for unobserved heterogeneity (see Wooldridge p.). Only 
        allows for one endogenous regressor: var_ls
        
        This method does one df and endogenous regressor.'''
        
    # Prelims
    vars_x = righthand_x.split(' + ')
    vars_z = righthand_z.split(' + ')
    
    num_z = righthand_z.count('+') + 1
    
    dep_var_step2 = ['net_coffratio_tot_ta','allowratio_tot_ta','rwata','provratio']
    
    #----------------------------------------------    
    # First check the data on column rank
    rank_full = np.linalg.matrix_rank(df[vars_x + vars_z]) 
    
    if rank_full != len(vars_x + vars_z):
        return([])
    
    #----------------------------------------------
    # STEP 1: First Stage
    #----------------------------------------------
                                
    # Estimate G_hat
    res_step1 = PanelOLS.from_formula(var_ls + ' ~ ' + righthand_x + ' + ' + righthand_z + ' + ' + time_dummies,\
                                      data = df).fit(cov_type = 'clustered', cluster_entity = True)
    
    df['G_hat_fd'] = res_step1.fitted_values

    #----------------------------------------------
    # Calculate G_hat_x_xbar for both first stages
    G_hat_x_xbar_fd = df.loc[:,df.columns.str.contains('_xbar')] * df.G_hat_fd[:, None]

    df[[x + '_G_hat' for x in vars_x]] = G_hat_x_xbar_fd
  
    #----------------------------------------------
    # Step 2: Second Stage
    #----------------------------------------------

    res_step2 = []
    for dep_var in dep_var_step2:
        # without correction unobserved heterogeneity
        res_step2a = PanelOLS.from_formula(dep_var + ' ~ ' + righthand_x +\
                     ' + ' + 'G_hat_fd' + ' + ' + time_dummies, data = df).\
                     fit(cov_type = 'clustered', cluster_entity = True)
        res_step2.append(res_step2a)
        
        # with correction unobserved heterogeneity
        res_step2b = PanelOLS.from_formula(dep_var + ' ~ ' + righthand_x +\
                     ' + ' + 'G_hat_fd' + '+' + righthand_ghat + ' + ' + time_dummies, data = df).\
                     fit(cov_type = 'clustered', cluster_entity = True)
        res_step2.append(res_step2b)
 
    #----------------------------------------------
    # Tests
    '''We test for three things:
        1) The strength of the instrument using a DWH test. F-stat must be > 10.
        2) A test whether dum_ls or ls_tot_ta is endogenous. H0: variable is exogenous
        3) A Sargan test to test the overidentifying restrictions. H0: overidentifying restrictions hold
        
        For only one instrument we use the p-val of the instrument in step 1) and we do not do a Sargan test'''        
    #----------------------------------------------
    
    #----------------------------------------------                                
    ## Weak instruments
    
    if num_z == 1:
        f_test_step1 = res_step1.pvalues[vars_z]
    else:
        res_step1b = PanelOLS.from_formula(var_ls + ' ~ ' + righthand_x + ' + ' + time_dummies, data = df).fit(cov_type = 'clustered', cluster_entity = True)
        f_test_step1 = fTestWeakInstruments(df[var_ls], res_step1.fitted_values, res_step1b.fitted_values, num_z)
    
    #----------------------------------------------
    ## Endogenous loan sales variable
    
    df['resid_step1'] = res_step1.resids

    pvals_ls_endo = []
    for dep_var in dep_var_step2:
        res_endo = PanelOLS.from_formula(dep_var + ' ~ ' + righthand_x +\
                       ' + ' + 'dum_ls' + ' + ' + 'resid_step1' + ' + ' + time_dummies, data = df).\
                       fit(cov_type = 'clustered', cluster_entity = True)
        pvals_ls_endo.append(res_endo.pvalues['resid_step1'])
              
    #----------------------------------------------
    ## Sargan test
    
    if num_z == 1:
        return(res_step1,res_step2,f_test_step1,pvals_ls_endo,[])
    else:
        sargan_res = []
        
        for model in res_step2:
            oir = sargan(model.resids, df[righthand_x.split(' + ')], df[righthand_z.split(' + ')])
            
            sargan_res.append(oir.pval)
        
        return(res_step1,res_step2,f_test_step1,pvals_ls_endo,sargan_res)

#----------------------------------------------
def tableIVtests(num_models,f_test_step1,pvals_ls_endo,sargan_res = None):
    '''Method to create a nice table for the test result of the FDIV'''
    # Create a list for f-test pvals_ls_endo that are the same length of number of models
    f_test_list = [f_test_step1] * len(num_models)
    pvals_l_endo_list = [j for i in zip(pvals_ls_endo,pvals_ls_endo) for j in i]
    if sargan_res:
        indicator_tests = [a*b*c for a,b,c in zip([(i > 10) * 1 for i in f_test_list],\
                          [(i < 0.05) * 1 for i in pvals_l_endo_list],\
                          [(i > 0.05) * 1 for i in sargan_res])]
        index = ['F-test weak instruments','P-val endogenous w','P-val Sargan','Indicator']
        columns = ['charge_2a','charge_2b','allow_2a','allow_2b',\
                        'rwata_2a','rwata_2b','prov_2a','prov_2b',]
        
        return(pd.DataFrame([f_test_list,pvals_l_endo_list,sargan_res,indicator_tests],\
                             index = index, columns = columns))
    else:
        indicator_tests = [a*b*c for a,b,c in zip([(i > 10) * 1 for i in f_test_list],\
                          [(i < 0.05) * 1 for i in pvals_l_endo_list])]
        index = ['T-test weak instruments','P-val endogenous w','Indicator']
        columns = ['charge_2a','charge_2b','allow_2a','allow_2b',\
                        'rwata_2a','rwata_2b','prov_2a','prov_2b',]
        
        return(pd.DataFrame([f_test_list,pvals_l_endo_list,indicator_tests],\
                             index = index, columns = columns))        
#----------------------------------------------       
def scoreFDIVtest(test_table):
    '''Makes a test score by summing over all the indicators and dividing that
        number by the number of models'''
    num_cols = test_table.shape[1]
    return(np.sum(test_table.Indicator) / num_cols)
            
#----------------------------------------------
#----------------------------------------------
# Analyses
#----------------------------------------------
#----------------------------------------------

# Set the righthand side of the formulas
if log:
    righthand_x = r'RC7205 + loanratio + roa + depratio + comloanratio + RC2170 + bhc'
    righthand_ghat_w = r'RC7205_G_hat + loanratio_G_hat + roa_G_hat + depratio_G_hat + comloanratio_G_hat + RC2170_G_hat + bhc_G_hat'
    righthand_ghat_ls = r'RC7205_G_hat + loanratio_G_hat + roa_G_hat + depratio_G_hat + comloanratio_G_hat + RC2170_G_hat + bhc_G_hat'
else:
    righthand_x = r'RC7205 + loanratio + roa + depratio + comloanratio + size + bhc'
    righthand_ghat_w = r'RC7205_G_hat + loanratio_G_hat + roa_G_hat + depratio_G_hat + comloanratio_G_hat + size_G_hat + bhc_G_hat'
    righthand_ghat_ls = r'RC7205_G_hat + loanratio_G_hat + roa_G_hat + depratio_G_hat + comloanratio_G_hat + size_G_hat + bhc_G_hat' 

vars_endo = ['dum_ls','ls_tot_ta'] 

# Calculate all possible combinations in vars_z
vars_z = ['num_branch','RIAD4150','perc_limited_branch','perc_full_branch','unique_states']

vars_z_comb = []
for L in range(1, len(vars_z)+1):
    for subset in itertools.combinations(vars_z, L):
        vars_z_comb.append(' + '.join(list(subset)))
    
# Setup up th rest of the data for loops
list_dfs = [df_full_fd, df_pre_fd, df_during_fd ,df_post_fd]
list_ghat = [righthand_ghat_w, righthand_ghat_ls]

#----------------------------------------------
# Run models
#----------------------------------------------

# Setup the lists
res_step1 = []
res_step2 = []
f_test_step1 = []
pvals_ls_endo = []
sargan_res = []

for data in list_dfs:
    time_dummies = data.columns[data.columns.str.contains('dum2')][1:]
    for i in range(len(vars_endo)):
        for z in vars_z_comb:
            res_step1_load,res_step2_load,f_test_step1_load,pvals_ls_endo_load,sargan_res_load =\
            analysesFDIV(data, vars_endo[i], righthand_x, list_dfs[i], z,  time_dummies)
                
            res_step1.append(res_step1_load)
            res_step2.append(res_step2_load)
            f_test_step1.append(f_test_step1_load)
            pvals_ls_endo.append(pvals_ls_endo_load)
            sargan_res.append(sargan_res_load)

#----------------------------------------------
# Make test tables and scoring vector
#----------------------------------------------
test_tables = []

for f, endo, oir in zip(f_test_step1,pvals_ls_endo,sargan_res):
    test_table_load = tableIVtests(8,f,endo,oir)
    test_tables.append(test_table_load)

scoring_vector = []
for table in test_tables:
    score = scoreFDIVtest(table)
    scoring_vector.append(score)

#----------------------------------------------
# Save the tables and scoring vector
#----------------------------------------------
# Make a vector containing names for the sheets
list_steps = ['_step1','_step2','_tests']
names_dfs = ['Full_','Pre_','During_','Post_']
iter_z = ['{}'.format(i) for i in range(len(vars_z_comb))]
names_dfs_endo = [(a + b) for a in names_dfs for b in vars_endo] 
names_dfs_endo_num = [(a + b) for a in names_dfs_endo for b in iter_z] 
sheet_names = [(a + b) for a in names_dfs_endo_num for b in list_steps]
 
#----------------------------------------------
if log:
    path = r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char\FD_IV_v2_log.xlsx'
else:
    path = r'X:\My Documents\PhD\Materials_dissertation\2-Chapter_2-bank_char\FD_IV_v2.xlsx'

# TODO: rename index

with pd.ExcelWriter(path) as writer:
    for i in len(res_step1):
        res_step1[i].to_excel(writer, sheet_name = sheet_names[i])
        res_step2[i].to_excel(writer, sheet_name = sheet_names[i+1])
        test_tables[i].to_excel(writer, sheet_name = sheet_names[i+2])

        